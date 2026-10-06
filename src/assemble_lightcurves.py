import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
import astropy.units as u

_ROOT = Path(__file__).resolve().parents[1]

def read_bandpass(fname):
   tbl_bp = Table.read(str(_ROOT / "data" / fname), format='ascii')
   wcol, tcol = tbl_bp.colnames[:2]
   w = np.asarray(tbl_bp[wcol], dtype=float)
   T = np.asarray(tbl_bp[tcol], dtype=float)
   try:
       unit = tbl_bp[wcol].unit
       if unit is not None:
           w = (w * unit).to('angstrom').value
   except Exception:
       pass
   return w, T


_C_AA = 2.99792458e18  # A/s
_F0_AB = 3.631e-20     # erg/s/cm^2/Hz

def abmag(w_obs, f_obs, wf, tf):
   # photon-counting: response weights photon rate, not energy flux
   mask = (wf > w_obs.min()) & (wf < w_obs.max())
   if mask.sum() < 5:
       return np.nan
   # fine uniform grid: tabulated curves too coarse for lines and the Balmer break
   lam = np.linspace(wf[mask].min(), wf[mask].max(), 8001)
   T = np.interp(lam, wf, tf)
   f_lam = np.interp(lam, w_obs, f_obs)
   num = np.trapezoid(f_lam * T * lam, lam)
   den = _F0_AB * _C_AA * np.trapezoid(T / lam, lam)
   if (num <= 0) or (den <= 0):
       return np.nan
   return -2.5 * np.log10(num / den)


def ab_minus_vega(wf, tf):
   # CALSPEC vacuum wavelengths -> air, as the passbands
   d = fits.getdata(_ROOT / "data/alpha_lyr_stis_012.fits", 1)
   w = np.asarray(d["WAVELENGTH"], dtype=float)
   s2 = (1e4 / w) ** 2
   w = w / (1 + 8.34254e-5 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))
   return abmag(w, np.asarray(d["FLUX"], dtype=float),
                np.asarray(wf, dtype=float), np.asarray(tf, dtype=float))


def serc_to_sdss_lc(
   df_lc: pd.DataFrame,
   z_col: str = "z",
   objid_col: str = "OBJID",
   band_col: str = "band",
   mag_col: str = "SMAG",
   out_mag_col: str = "MAG_SDSS",
   out_band_col: str = "band_sdss",
   survey_col: str = "survey",
   template_path: str = str(_ROOT / "data/vandenberk_qso_composite_cds.txt"),
) -> pd.DataFrame:
   # natural plate passbands, Vega-based (AB-Vega + AB synthetic colour); POSS-I uses 103a-E
   bp_files = {'g': 'SLOAN_SDSS.g.dat', 'r': 'SLOAN_SDSS.r.dat', 'i': 'SLOAN_SDSS.i.dat',
               'BJ': 'serc-j.txt', 'R': 'serc-r.txt', 'E': 'possi-e.txt', 'I': 'serc-i.txt'}
   bands = {b: read_bandpass(fname) for b, fname in bp_files.items()}
   vega = {b: ab_minus_vega(*bands[b]) for b in ('BJ', 'R', 'E', 'I')}

   required = {objid_col, z_col, band_col, mag_col, survey_col}
   missing = required - set(df_lc.columns)
   if missing:
       raise ValueError(f"df_lc is missing required columns: {sorted(missing)}")

   tbl = Table.read(template_path, format="ascii.cds")
   wave_rest = np.asarray(tbl["Wave"], dtype=float)   # Angstrom
   flux_rest = np.asarray(tbl["FluxD"], dtype=float)  # relative flux density

   z_per_obj = df_lc.groupby(objid_col, sort=False)[z_col].first()

   uniq_z = pd.unique(z_per_obj.to_numpy())

   m_cache = {}
   for z in uniq_z:
       if not np.isfinite(z):
           continue
       lam_obs = wave_rest * (1.0 + z)
       f_obs = flux_rest / (1.0 + z)

       mz = {}
       for key in ("BJ", "R", "E", "I", "g", "r", "i"):
           wf, tf = bands[key]
           mz[key] = abmag(lam_obs, f_obs, np.asarray(wf, dtype=float), np.asarray(tf, dtype=float))

       if any(np.isnan(v) for v in mz.values()):
           m_cache[z] = (np.nan, np.nan, np.nan, np.nan)
       else:
           m_cache[z] = (mz["g"] - mz["BJ"] + vega["BJ"], mz["r"] - mz["R"] + vega["R"],
                         mz["r"] - mz["E"] + vega["E"], mz["i"] - mz["I"] + vega["I"])

   offsets = pd.DataFrame({objid_col: z_per_obj.index, z_col: z_per_obj.values})
   dg_arr, dr_arr, de_arr, di_arr = (np.full(len(offsets), np.nan) for _ in range(4))

   z_vals = offsets[z_col].to_numpy()
   for i, z in enumerate(z_vals):
       if np.isfinite(z) and (z in m_cache):
           dg_arr[i], dr_arr[i], de_arr[i], di_arr[i] = m_cache[z]

   offsets["dg"] = dg_arr
   offsets["dr"] = dr_arr
   offsets["de"] = de_arr
   offsets["di"] = di_arr

   out = df_lc.copy()
   out = out.merge(offsets[[objid_col, "dg", "dr", "de", "di"]], on=objid_col, how="left")
   possi = (out[survey_col].astype(str) == "sss_possi").to_numpy()
   out.loc[possi, "dr"] = out.loc[possi, "de"].to_numpy()

   out[out_mag_col] = np.nan
   out[out_band_col] = None

   band_vals = out[band_col].astype(str).to_numpy()
   mags = pd.to_numeric(out[mag_col], errors="coerce").to_numpy()

   m_bj = band_vals == "B_J"
   m_r  = band_vals == "R"
   m_i  = band_vals == "I"

   ok_bj = m_bj & np.isfinite(mags) & np.isfinite(out["dg"].to_numpy())
   ok_r  = m_r  & np.isfinite(mags) & np.isfinite(out["dr"].to_numpy())
   ok_i  = m_i  & np.isfinite(mags) & np.isfinite(out["di"].to_numpy())

   out.loc[ok_bj, out_mag_col] = mags[ok_bj] + out.loc[ok_bj, "dg"].to_numpy()
   out.loc[ok_r,  out_mag_col] = mags[ok_r]  + out.loc[ok_r,  "dr"].to_numpy()
   out.loc[ok_i,  out_mag_col] = mags[ok_i]  + out.loc[ok_i,  "di"].to_numpy()

   out.loc[ok_bj, out_band_col] = "g"
   out.loc[ok_r,  out_band_col] = "r"
   out.loc[ok_i,  out_band_col] = "i"

   out = out.drop(columns=["dg", "dr", "de", "di"], errors="ignore")

   return out



filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}
bands_sdss = ["u", "g", "r", "i", "z"]

def concat_light_curves_df(
   filter_object_ids=None,
   skip=None,
   N=None,
   data_dir="data/S82",
):
   cat = pd.read_parquet(f"{data_dir}/Catalog.parquet").set_index("idx")

   sdss = pd.read_parquet(f"{data_dir}/dr16s82_sdssLCRaw.parquet")
   ps1 = pd.read_parquet(f"{data_dir}/dr16s82_ps1LCRaw.parquet")
   ztf = pd.read_parquet(f"{data_dir}/dr16s82_ZuberLCRaw.parquet")

   sdss = sdss.loc[sdss["mjd"].notna()]

   sdss_objids = pd.Index(sdss["objectId"].unique())
   if filter_object_ids is not None:
       sdss_objids = sdss_objids.intersection(pd.Index(filter_object_ids))

   cat = cat.loc[cat["objectId"].isin(sdss_objids)]
   if skip is not None:
       cat = cat.iloc[skip:]
   if N is not None:
       cat = cat.iloc[:N]

   if len(cat) == 0:
       return pd.DataFrame(columns=["objectId", "band", "time", "mag", "magerr", "survey"])

   cat_keys = cat.reset_index(drop=False).copy()
   cat_keys["ps1objID"] = cat_keys["ps1objID"].astype("Int64")

   offset_wide = pd.DataFrame(
       {
           "objectId": cat_keys["objectId"].values,
           "g": cat_keys["sdss_g_qg"].values - cat_keys["ps1_g_qg"].values,
           "r": cat_keys["sdss_r_qg"].values - cat_keys["ps1_r_qg"].values,
           "i": cat_keys["sdss_i_qg"].values - cat_keys["ps1_i_qg"].values,
           "z": cat_keys["sdss_z_qg"].values - cat_keys["ps1_z_qg"].values,
       }
   )
   offset_long = offset_wide.melt(id_vars=["objectId"], var_name="band", value_name="offset")
   u_offsets = pd.DataFrame({"objectId": cat_keys["objectId"].values, "band": "u", "offset": 0.0})
   offset_long = pd.concat([offset_long, u_offsets], ignore_index=True)
   offset_long["offset"] = pd.to_numeric(offset_long["offset"], errors="coerce").fillna(0.0)

   band_map = pd.DataFrame({"band": bands_sdss, "filterID": [filters[b] for b in bands_sdss]})

   sdss_df = (
       sdss.loc[sdss["objectId"].isin(cat_keys["objectId"])]
       .merge(band_map, on="filterID", how="inner")
       .loc[:, ["objectId", "band", "mjd", "psMag", "psMagErr_p3"]]
       .rename(columns={"mjd": "time", "psMag": "mag", "psMagErr_p3": "magerr"})
   )
   sdss_df["survey"] = "sdss"

   ps1_df = (
       ps1.merge(cat_keys.loc[:, ["ps1objID", "objectId"]], on="ps1objID", how="inner")
       .merge(band_map, on="filterID", how="inner")
       .loc[:, ["objectId", "band", "obsTime", "psfMag", "psfMagErr_p3"]]
       .rename(columns={"obsTime": "time", "psfMag": "mag", "psfMagErr_p3": "magerr"})
   )
   ps1_df["survey"] = "ps1"

   ztf_df = (
       ztf.merge(cat_keys.loc[:, ["ps1objID", "objectId"]], on="ps1objID", how="inner")
       .merge(band_map, on="filterID", how="inner")
       .loc[:, ["objectId", "band", "mjd", "mag", "magerr_p3"]]
       .rename(columns={"mjd": "time", "magerr_p3": "magerr"})
   )
   ztf_df["survey"] = "ztf"

   ps1_df = ps1_df.merge(offset_long, on=["objectId", "band"], how="left")
   ztf_df = ztf_df.merge(offset_long, on=["objectId", "band"], how="left")
   ps1_df["mag"] = ps1_df["mag"] + ps1_df["offset"].fillna(0.0)
   ztf_df["mag"] = ztf_df["mag"] + ztf_df["offset"].fillna(0.0)
   ps1_df = ps1_df.drop(columns=["offset"])
   ztf_df = ztf_df.drop(columns=["offset"])

   df = pd.concat([sdss_df, ps1_df, ztf_df], ignore_index=True)

   for col in ["time", "mag", "magerr"]:
       df[col] = pd.to_numeric(df[col], errors="coerce")
   df = df.loc[df["time"].notna() & df["mag"].notna() & df["magerr"].notna()]
   df = df.loc[df["band"].isin(bands_sdss)]
   df = df.sort_values(["objectId", "band", "time"], kind="mergesort").reset_index(drop=True)
   return df


def sss_raw() -> pd.DataFrame:
   cache = _ROOT / "temp/sss_raw.parquet"
   if not cache.exists():
       cache.parent.mkdir(exist_ok=True)
       load_sss_supercosmos_raw(str(_ROOT / "data/wu_qso.hdf5"), str(_ROOT / "data/survey_table.fits")).to_parquet(cache)
   return pd.read_parquet(cache)


def load_sss_supercosmos_raw(wu_qso_h5: str, survey_table_fits: str) -> pd.DataFrame:
   with h5py.File(wu_qso_h5) as f:
       arr = np.concatenate([f[k][:] for k in f.keys()])
   df = pd.DataFrame({n: arr[n].astype(arr[n].dtype.newbyteorder("=")) for n in arr.dtype.names})
   df.columns = [c.upper() for c in df.columns]

   dat = fits.getdata(survey_table_fits)
   t = Table(dat)
   for name in t.colnames:
       if len(t[name].shape) > 1 and t[name].shape[1] == 1:
           t[name] = t[name].flatten()
   df_survey = t.to_pandas()
   df_survey.columns = [c.upper() for c in df_survey.columns]
   for c in df_survey.columns:
       if df_survey[c].dtype.kind in "iuf":
           df_survey[c] = df_survey[c].to_numpy().astype(df_survey[c].dtype.newbyteorder("="))

   if "PLATEID" not in df.columns:
       raise KeyError("wu_qso_h5 missing PLATEID")
   df = df.merge(df_survey, on="PLATEID", how="left")

   for cands, out in [
       (["RA", "ALPHA_J2000", "ALPHA"], "RA"),
       (["DEC", "DELTA_J2000", "DELTA"], "DEC"),
       (["MJD", "TIME", "T"], "MJD"),
       (["SMAG", "MAG", "MAGNITUDE"], "SMAG"),
   ]:
       if out not in df.columns:
           for c in cands:
               if c in df.columns:
                   df[out] = df[c]
                   break
   if "SMAG_ERR" not in df.columns:
       for c in ["SMAG_ERR", "MAGERR", "MAG_ERR", "ERR"]:
           if c in df.columns:
               df["SMAG_ERR"] = df[c]
               break
   if "SMAG_ERR" not in df.columns:
       df["SMAG_ERR"] = np.nan

   for c in ["RA", "DEC", "MJD", "SMAG", "SMAG_ERR"]:
       df[c] = pd.to_numeric(df[c], errors="coerce")
   df = df.dropna(subset=["RA", "DEC", "MJD", "SMAG"]).reset_index(drop=True)

   if "SURVEYNAME" not in df.columns:
       raise KeyError("survey table merge did not provide SURVEYNAME")
   df["SURVEYNAME"] = df["SURVEYNAME"].astype(str)
   df["BAND_NATIVE"] = df["SURVEYNAME"].map(norm_native_from_surveyname)
   if not df["BAND_NATIVE"].isin(["B_J", "R", "I"]).any():
       raise ValueError("no SSS rows mapped to B_J/R/I bands")
   return df


class _UnionFind:
   def __init__(self, n: int):
       self.parent = np.arange(n, dtype=int)
       self.rank = np.zeros(n, dtype=int)

   def find(self, a: int) -> int:
       while self.parent[a] != a:
           self.parent[a] = self.parent[self.parent[a]]
           a = self.parent[a]
       return a

   def union(self, a: int, b: int):
       ra = self.find(a)
       rb = self.find(b)
       if ra == rb:
           return
       if self.rank[ra] < self.rank[rb]:
           self.parent[ra] = rb
       elif self.rank[ra] > self.rank[rb]:
           self.parent[rb] = ra
       else:
           self.parent[rb] = ra
           self.rank[ra] += 1

def assign_group_ids_by_sky(df: pd.DataFrame, radius_arcsec: float = 1.0) -> pd.DataFrame:
   out = df.copy().reset_index(drop=True)
   coords = SkyCoord(ra=out["RA"].to_numpy() * u.deg,
                     dec=out["DEC"].to_numpy() * u.deg,
                     frame="icrs")

   i1, i2, _, _ = coords.search_around_sky(coords, seplimit=radius_arcsec * u.arcsec)
   uf = _UnionFind(len(out))
   for a, b in zip(i1, i2):
       if int(a) != int(b):
           uf.union(int(a), int(b))

   roots = np.array([uf.find(i) for i in range(len(out))], dtype=int)
   _, gid = np.unique(roots, return_inverse=True)
   out["GROUP_ID"] = gid.astype(int)
   return out


def drop_blended_parents(df: pd.DataFrame) -> pd.DataFrame:
   # one detection per (group, plate): deblended child, then nearest centroid
   cen = df.groupby("GROUP_ID")[["RA", "DEC"]].transform("median")
   dra = (df["RA"] - cen["RA"]) * np.cos(np.deg2rad(df["DEC"]))
   sep = np.hypot(dra, df["DEC"] - cen["DEC"])
   out = df.assign(_parent=(df["SOURCEID"] == 0).astype(int), _sep=sep)
   out = out.sort_values(["_parent", "_sep"], kind="mergesort")
   out = out.drop_duplicates(subset=["GROUP_ID", "PLATEID"], keep="first")
   return out.drop(columns=["_parent", "_sep"]).sort_index()


SATURATION_LIMITS = {"B_J": 16.0, "R": 14.5, "R1": 15.0, "I": 14.0}


def saturation_mask(smag: np.ndarray, native_band: np.ndarray) -> np.ndarray:
   lim = pd.Series(native_band).map(SATURATION_LIMITS).to_numpy(dtype=float)
   return np.asarray(smag, dtype=float) >= lim


def sss_to_sdss_longform(df_sss_with_objid: pd.DataFrame, bands_bp=None, sat_cut: bool = True) -> pd.DataFrame:
   df = df_sss_with_objid.copy()

   req = ["OBJID", "z", "MJD", "SMAG", "SMAG_ERR", "BAND_NATIVE", "SURVEYNAME", "PLATEID"]
   miss = [c for c in req if c not in df.columns]
   if miss:
       raise ValueError(f"SSS table missing columns: {miss}")

   df_in = pd.DataFrame({
       "OBJID": df["OBJID"].astype(str),
       "z": pd.to_numeric(df["z"], errors="coerce"),
       "band": df["BAND_NATIVE"].astype(str),
       "MJD": pd.to_numeric(df["MJD"], errors="coerce"),
       "SMAG": pd.to_numeric(df["SMAG"], errors="coerce"),
       "SMAG_ERR": pd.to_numeric(df["SMAG_ERR"], errors="coerce"),
       "survey": np.where(df["SURVEYNAME"].astype(str) == "POSSI-E(S)", "sss_possi", "sss"),
       "PLATEID": pd.to_numeric(df["PLATEID"], errors="coerce"),
       "calib_ok": df["calib_ok"].to_numpy(bool) if "calib_ok" in df.columns else True,
       "trunc_v": df["trunc_v"].to_numpy(float) if "trunc_v" in df.columns else 1.0,
   }).dropna(subset=["OBJID", "z", "band", "MJD", "SMAG", "PLATEID"])

   df_in = df_in[df_in["band"].isin(["B_J", "R", "I"])].copy()

   if sat_cut:
       nat = np.where(df_in["survey"].to_numpy() == "sss_possi", "R1", df_in["band"].to_numpy())
       keep = saturation_mask(df_in["SMAG"].to_numpy(dtype=float), nat)
       print(f"saturation cut: dropped {int((~keep).sum())} plate epochs of {len(keep)} over "
             f"{df_in.loc[~keep, 'OBJID'].nunique()} objects")
       df_in = df_in[keep].copy()

   have_converter = ("serc_to_sdss_lc" in globals()) and callable(globals()["serc_to_sdss_lc"])

   if have_converter:
       df_conv = globals()["serc_to_sdss_lc"](df_in)

       for c in ["OBJID", "MJD", "band_sdss", "MAG_SDSS"]:
           if c not in df_conv.columns:
               raise ValueError(f"serc_to_sdss_lc output missing required column: {c}")

       out = pd.DataFrame({
           "OBJID": df_conv["OBJID"].astype(str),
           "band": df_conv["band_sdss"].astype(str).str.lower(),
           "time": pd.to_numeric(df_conv["MJD"], errors="coerce"),
           "mag": pd.to_numeric(df_conv["MAG_SDSS"], errors="coerce"),
           "magerr": pd.to_numeric(df_conv.get("SMAG_ERR", np.nan), errors="coerce"),
           "survey": df_conv["survey"].astype(str),
           "plate": pd.to_numeric(df_conv["PLATEID"], errors="coerce"),
           "smag": pd.to_numeric(df_conv["SMAG"], errors="coerce"),
           "calib_ok": df_conv["calib_ok"].to_numpy(bool),
           "trunc_v": df_conv["trunc_v"].to_numpy(float),
       }).dropna(subset=["OBJID", "band", "time", "mag"])

       return out

   warnings.warn(
       "serc_to_sdss_lc() not found. Falling back to B_J->g, R->r, I->i with mag=SMAG (not a true conversion).",
       RuntimeWarning
   )
   native_to_sdss = {"B_J": "g", "R": "r", "I": "i"}
   out = pd.DataFrame({
       "OBJID": df_in["OBJID"].astype(str),
       "band": df_in["band"].map(native_to_sdss).astype(str),
       "time": df_in["MJD"].astype(float),
       "mag": df_in["SMAG"].astype(float),
       "magerr": df_in["SMAG_ERR"].astype(float),
       "survey": df_in["survey"].astype(str),
       "plate": df_in["PLATEID"].astype(float),
       "smag": df_in["SMAG"].astype(float),
       "calib_ok": df_in["calib_ok"].to_numpy(bool),
       "trunc_v": df_in["trunc_v"].to_numpy(float),
   }).dropna(subset=["OBJID", "band", "time", "mag"])
   return out


def assemble_total_lightcurve(df_base: pd.DataFrame, df_sss: pd.DataFrame) -> pd.DataFrame:
   df = pd.concat([df_base, df_sss], ignore_index=True, sort=False)

   df["OBJID"] = df["OBJID"].astype(str)
   df["band"] = df["band"].astype(str).str.lower()
   for c in ["time", "mag", "magerr"]:
       df[c] = pd.to_numeric(df[c], errors="coerce")
   df = df.dropna(subset=["OBJID", "band", "time", "mag"])

   df = df[df["band"].isin(["u", "g", "r", "i", "z"])].copy()
   if "plate" not in df.columns:
       df["plate"] = -1
   df["plate"] = pd.to_numeric(df["plate"], errors="coerce").fillna(-1).astype(np.int64)
   df["calib_ok"] = df["calib_ok"].fillna(True).astype(bool)
   df["trunc_v"] = (pd.to_numeric(df["trunc_v"], errors="coerce").fillna(1.0).astype(float)
                    if "trunc_v" in df.columns else 1.0)
   df = df.drop_duplicates(subset=["OBJID", "band", "time", "mag", "survey"])
   df = df.sort_values(["OBJID", "band", "time"], kind="mergesort").reset_index(drop=True)
   return df

def norm_native_from_surveyname(x) -> str:
   s = str(x).strip().upper().replace(" ", "")
   if ("SERC-J" in s) or ("/EJ" in s) or s.endswith("-J") or ("SERCJ" in s):
       return "B_J"
   if ("SERC-R" in s) or ("AAO-R" in s) or s.endswith("-R") or ("SERCR" in s) or (s == "POSSI-E(S)"):
       return "R"
   if ("SERC-I" in s) or s.endswith("-I") or (s == "I") or ("SERCI" in s):
       return "I"
   return s


_SURVEY_CODE_BAND = {1: "B_J", 2: "R", 3: "I"}
_CALIB_BINS = np.arange(14.0, 21.0, 0.5)
_CALIB_MIN_N = 50
_R1_BINS = np.arange(14.6, 20.01, 0.2)
_R1_MIN_N = 300
_R1_DEG = 4

def _mad_sigma(x: np.ndarray) -> float:
   return 1.4826 * np.median(np.abs(x - np.median(x)))

def _populated_bins(vals: np.ndarray):
   idx = np.digitize(vals, _CALIB_BINS)
   for bi in range(1, len(_CALIB_BINS)):
      sel = idx == bi
      if int(sel.sum()) >= _CALIB_MIN_N:
         yield (_CALIB_BINS[bi - 1] + _CALIB_BINS[bi]) / 2, sel


def build_sss_calibration(out_csv: str | None, zp: pd.DataFrame | None = None, pin: bool = True,
                          keep: np.ndarray | None = None, bins: np.ndarray = _R1_BINS,
                          deg: int = _R1_DEG, ext: float = 0.0) -> pd.DataFrame:
   # quasar cuts on standards; R1 offset detections stay uncut (cut is on calibrated R1)
   det = drop_blended_parents(star_detections_cached().rename(columns=_STAR_BLEND)).rename(
      columns={v: k for k, v in _STAR_BLEND.items()})
   if keep is not None:
      det = det[keep[det["star"].to_numpy()]]
   possi = det["possi"].to_numpy()
   raw = det["SMAG"].to_numpy(dtype=float)
   nat = np.where(possi, "R1", det["bi"].map(_SSS_BI_BAND).to_numpy())
   mag = raw if zp is None else raw - plate_zp_offset(
      zp, det["plate"].to_numpy(), det["ra"].to_numpy(float), det["dec"].to_numpy(float))[0]
   s = pd.DataFrame({"star": det["star"].to_numpy(), "mag": mag, "raw": raw,
                     "code": np.where(possi, 9, det["bi"].to_numpy() + 1)})
   s = s[saturation_mask(raw, nat) | possi].reset_index(drop=True)

   g = s.groupby(["star", "code"], sort=True)["mag"]
   rep = pd.DataFrame({
      "med": g.median(), "n": g.size(),
      "mad": 1.4826 * (s["mag"] - g.transform("median")).abs()
      .groupby([s["star"], s["code"]]).median(),
   }).reset_index()
   rep = rep[rep["n"] >= 2]

   rows = []
   for code, band in _SURVEY_CODE_BAND.items():
      r = rep[rep["code"] == code]
      mad = r["mad"].to_numpy()
      for center, sel in _populated_bins(r["med"].to_numpy()):
         rows.append({"band": band, "mag_center": center, "offset": 0.0, "offset_err": 0.0,
                      "sigma": float(np.median(mad[sel])), "n_stars": int(sel.sum())})
   calib = pd.DataFrame(rows)

   # R1 -> R2 map instrumented by Ivezic CCD r, since conditioning on R1 shrinks to the mean;
   # per r bin median R2 and R1-R2, root-n weighted quartic in R2, inverted
   med = s.groupby(["star", "code"])[["mag", "raw"]].agg(["median", "size"])
   m2, m1 = med.xs(2, level="code"), med.xs(9, level="code")
   both = m1.index.intersection(m2.index)
   R1 = m1[("mag", "median")].reindex(both).to_numpy()
   R2 = m2[("mag", "median")].reindex(both).to_numpy()
   D_raw = (m1[("raw", "median")].reindex(both).to_numpy()
            - m2[("raw", "median")].reindex(both).to_numpy())
   n_r2 = m2[("mag", "size")].reindex(both).to_numpy(dtype=float)
   n_r1 = m1[("mag", "size")].reindex(both).to_numpy(dtype=float)
   ivz, _, npos = ivezic_star_mags()
   ref_mags = pd.Series(ivz["r"], index=np.arange(npos)).reindex(both).to_numpy()
   ok = np.isfinite(ref_mags)
   R1, R2, D_raw, n_r2, n_r1, ref_mags = (v[ok] for v in (R1, R2, D_raw, n_r2, n_r1, ref_mags))
   if len(R1) < 300_000:
      raise ValueError(f"R1/R2 calibrator count {len(R1)} < 300000")

   r2_curve = calib[calib["band"] == "R"].set_index("mag_center")["sigma"]
   nodes = []
   for lo, hi in zip(bins[:-1], bins[1:]):
      sel = (ref_mags >= lo) & (ref_mags < hi)
      if int(sel.sum()) < _R1_MIN_N:
         continue
      t, d = float(np.median(R2[sel])), R1[sel] - R2[sel]
      sigma_tot = _mad_sigma(d)
      # csv R sigma = sigma/sqrt(2) (MAD of 2-epoch repeats); R2 is a median over n_r2 epochs
      sigma_r2 = float(np.interp(t, r2_curve.index.to_numpy(), r2_curve.to_numpy()))
      var = sigma_tot ** 2 - 2.0 * sigma_r2 ** 2 / float(np.median(n_r2[sel]))
      floor = (0.5 * sigma_tot) ** 2
      if var < floor:
         warnings.warn(f"sigma_R1 floor applied at R2 {t:.2f}: var {var:.4f} < floor {floor:.4f}")
         var = floor
      nodes.append((t, float(np.median(d)), float(np.sqrt(var)),
                    1.2533 * sigma_tot / np.sqrt(sel.sum()), int(sel.sum()),
                    float(np.median(D_raw[sel]))))
   if len(nodes) < 10:
      raise ValueError(f"only {len(nodes)} populated R1 offset bins")
   t, d, sig, derr, n, d_raw = (np.array(v) for v in zip(*nodes))
   # constant degenerate with POSS-I zero points; level pinned on the zero-point-free raw differential
   if pin:
      d = d - np.average(d, weights=n) + np.average(d_raw, weights=n)
   design = lambda x: np.vstack([np.asarray(x, float) ** k for k in range(deg + 1)]).T
   print(f"R1 map level: unpinned {np.average(d, weights=n) - np.average(d_raw, weights=n):+.4f} "
         f"mag against the raw differential")
   fold = np.arange(len(R1)) % 5
   for cond, key in [("CCD r", ref_mags), ("R1", R1)]:
      res = np.zeros(len(R1))
      for f in range(5):
         tr = fold != f
         nod = [(np.median((R1 if cond == "R1" else R2)[s]), np.median(R1[s] - R2[s]), int(s.sum()))
                for lo, hi in zip(bins[:-1], bins[1:])
                for s in [tr & (key >= lo) & (key < hi)] if int(s.sum()) >= _R1_MIN_N]
         tt, dd, nn = (np.array(v) for v in zip(*nod))
         c = np.linalg.lstsq(design(tt) * np.sqrt(nn)[:, None], dd * np.sqrt(nn), rcond=None)[0]
         gr = np.arange(tt.min(), tt.max() + 1e-9, 0.05)
         og = design(gr) @ c
         res[~tr] = R1[~tr] - np.interp(R1[~tr], gr if cond == "R1" else gr + og, og) - R2[~tr]
      bm, bw = (np.array(v) for v in zip(*[(np.median(res[s]), n_r1[s].sum())
                                          for lo, hi in zip(bins[:-1], bins[1:])
                                          for s in [(ref_mags >= lo) & (ref_mags < hi)]
                                          if int(s.sum()) >= _R1_MIN_N]))
      print(f"R1 map held out five-fold, conditioned on {cond}: epoch-weighted residual of the "
            f"CCD-binned held-out medians {np.average(bm, weights=bw):+.4f} mag")
   coef = np.linalg.lstsq(design(t) * np.sqrt(n)[:, None], d * np.sqrt(n), rcond=None)[0]
   grid = np.arange(t.min(), t.max() + ext + 1e-9, 0.05)
   off = design(grid) @ coef
   if not np.all(np.diff(grid + off) > 0):
      raise ValueError("R1 offset map is not monotone over the fitted range")
   node_rms = float(np.sqrt(np.average((d - design(t) @ coef) ** 2, weights=n)))
   print(f"R1 map: {len(R1)} standards, {len(t)} nodes over R2 {t.min():.3f}-{t.max():.3f}, "
         f"weighted node rms {node_rms:.4f} mag, valid raw R1 "
         f"{grid[0] + off[0]:.3f}-{grid[-1] + off[-1]:.3f}")
   calib = pd.concat([calib, pd.DataFrame({
      "band": "R1", "mag_center": grid + off, "offset": off,
      "offset_err": np.interp(grid, t, derr), "sigma": np.interp(grid, t, sig),
      "n_stars": np.interp(grid, t, n).astype(int)})], ignore_index=True)

   if (calib["offset"].abs() > 1.0).any():
      raise ValueError("R1 offset exceeds 1.0 mag")
   if ((calib["sigma"] <= 0.01) | (calib["sigma"] >= 1.5)).any():
      raise ValueError("calibration sigma outside (0.01, 1.5) mag")

   calib = calib.sort_values(["band", "mag_center"]).reset_index(drop=True)
   if out_csv:
      calib.to_csv(out_csv, index=False)
   return calib

def apply_sss_calibration(df_sss: pd.DataFrame, calib: pd.DataFrame) -> pd.DataFrame:
   df = df_sss.copy()
   surveyname = df["SURVEYNAME"].astype(str)
   curves = {name: calib[calib["band"] == band].sort_values("mag_center")
             for name, band in [("SERC-J/EJ", "B_J"), ("SERC-R/AAO-R", "R"),
                                ("SERC-I", "I"), ("POSSI-E(S)", "R1")]}

   possi = (surveyname == "POSSI-E(S)").to_numpy()
   r1 = curves["POSSI-E(S)"]
   raw = df["SMAG"].to_numpy(dtype=float)
   mags = raw.copy()
   r1x = r1["mag_center"].to_numpy()
   mags[possi] -= np.interp(raw[possi], r1x, r1["offset"].to_numpy())
   df["SMAG"] = mags
   print(f"R1 map applied: median |shift| {np.median(np.abs(raw[possi] - mags[possi])):.4f} mag "
         f"over {int(possi.sum())} POSS-I detections")
   # outside R1 fit range: flagged, not clamped
   df["calib_ok"] = ~possi | ((raw >= r1x[0]) & (raw <= r1x[-1]))

   err = np.full(len(df), np.nan)
   for name, c in curves.items():
       m = (surveyname == name).to_numpy()
       err[m] = np.interp(raw[m], c["mag_center"].to_numpy(), c["sigma"].to_numpy())
   # csv B_J/R/I sigma = sigma/sqrt(2) (MAD of 2-epoch repeats); R1 sigma already single-epoch
   err[~possi] *= np.sqrt(2.0)
   df["SMAG_ERR_CAT"] = err
   df["SMAG_ERR"] = plate_epoch_sigma(raw, surveyname.to_numpy(), err)
   if not np.isfinite(df["SMAG_ERR"].to_numpy()).all():
       bad = surveyname[~np.isfinite(df["SMAG_ERR"].to_numpy())].unique()
       raise ValueError(f"non-finite SMAG_ERR after calibration for surveys: {list(bad)}")
   return df


_EPOCH_SIGMA_BAND = {"SERC-J/EJ": "B_J", "SERC-R/AAO-R": "R", "SERC-I": "I"}


def plate_epoch_sigma(raw: np.ndarray, surveyname: np.ndarray, cat: np.ndarray) -> np.ndarray:
   # star-curve sigma within its magnitude range; elsewhere and POSS-I keep table sigma
   c = pd.read_csv(_ROOT / "data/plate_star_error_check.csv")
   c = c[c["adopted"].to_numpy(bool)]
   out = np.asarray(cat, dtype=float).copy()
   for name, band in _EPOCH_SIGMA_BAND.items():
       d = c[c["band"] == band].sort_values("mag_lo")
       lo, hi = float(d["mag_lo"].min()), float(d["mag_hi"].max())
       m = (surveyname == name) & (raw >= lo) & (raw < hi)
       out[m] = np.interp(raw[m], 0.5 * (d["mag_lo"] + d["mag_hi"]).to_numpy(),
                          d["meas_scatter"].to_numpy())
   return out


_SSS_CODE_NAME = {1: "SERC-J/EJ", 2: "SERC-R/AAO-R", 3: "SERC-I", 9: "POSSI-E(S)"}
_SSS_CODE_BI = {1: 0, 2: 1, 3: 2, 9: 1}
_SSS_BI_BAND = {0: "B_J", 1: "R", 2: "I"}

def load_star_detections() -> pd.DataFrame:
   parts, stars = [], []
   with h5py.File(_ROOT / "data/roe_stars.hdf5") as f:
       for i, k in enumerate(f.keys()):
           d = f[k][:]
           parts.append(d)
           stars.append(np.full(len(d), i, dtype=np.int64))
   d = np.concatenate(parts)
   star = np.concatenate(stars)
   code = d["surveyID"].astype(np.int64)
   mag = d["sMag"].astype(float)
   ok = np.isfinite(mag) & np.isin(code, list(_SSS_CODE_NAME))
   star, code, mag, plate = star[ok], code[ok], mag[ok], d["plateID"][ok].astype(np.int64)
   sourceid, ra, dec = d["sourceID"][ok].astype(np.int64), d["ra"][ok], d["dec"][ok]

   with fits.open(_ROOT / "data/survey_table.fits") as h:
       t = h[1].data
       pid = np.asarray(t["PLATEID"]).ravel().astype(np.int64)
       mjd = np.asarray(t["MJD"]).ravel().astype(float)
       sname = np.asarray(t["SURVEYNAME"]).ravel().astype(str)
   order = np.argsort(pid)
   pos = np.searchsorted(pid, plate, sorter=order)
   if not np.array_equal(pid[order][pos], plate):
       raise ValueError("plateID missing from survey table")
   mjd_d, sname_d = mjd[order][pos], sname[order][pos]
   if (mjd_d <= 0).any():
       raise ValueError("non-positive MJD after plate join")
   for c, name in _SSS_CODE_NAME.items():
       if not (sname_d[code == c] == name).all():
           raise ValueError(f"surveyID {c} joined to SURVEYNAME other than {name}")
   bi = np.vectorize(_SSS_CODE_BI.get)(code)
   return pd.DataFrame({"star": star, "bi": bi, "mjd": mjd_d, "SURVEYNAME": sname_d,
                        "SMAG": mag.astype(np.float32), "possi": code == 9, "plate": plate,
                        "sourceID": sourceid, "ra": ra, "dec": dec})


def star_detections_cached() -> pd.DataFrame:
   cache = _ROOT / "temp/roe_star_detections.parquet"
   if cache.exists():
       df = pd.read_parquet(cache)
       if "sourceID" in df.columns:
           return df
   df = load_star_detections()
   cache.parent.mkdir(exist_ok=True)
   df.to_parquet(cache)
   return df


_STAR_BLEND = {"star": "GROUP_ID", "ra": "RA", "dec": "DEC", "sourceID": "SOURCEID", "plate": "PLATEID"}


def star_zp_detections(calib: pd.DataFrame, sat_cut: bool = True) -> pd.DataFrame:
   # quasar cuts on standards; raw keeps pre-calibration mag; sat_cut=False for error-model diagnostic
   det = (drop_blended_parents(star_detections_cached().rename(columns=_STAR_BLEND))
          .rename(columns={v: k for k, v in _STAR_BLEND.items()}).reset_index(drop=True))
   det = apply_sss_calibration(det.assign(raw=det["SMAG"].to_numpy(dtype=float)), calib)
   if not sat_cut:
      return det
   nat = np.where(det["possi"].to_numpy(), "R1", det["bi"].map(_SSS_BI_BAND).to_numpy())
   return det[saturation_mask(det["SMAG"].to_numpy(dtype=float), nat)].reset_index(drop=True)


def build_plate_zeropoints(det: pd.DataFrame, out_csv: str | None, min_stars: int = 30,
                           tol: float = 1e-3, max_iter: int = 200,
                           pos: dict[str, int] | None = None, det_limits: bool = True) -> pd.DataFrame:
   star = det["star"].to_numpy()
   bi = det["bi"].to_numpy()
   plate = det["plate"].to_numpy()
   mags = det["SMAG"].to_numpy(dtype=float).copy()
   multi = pd.Series(mags).groupby([star, bi]).transform("size").to_numpy() >= 2
   plates_all = np.unique(plate)
   n_stars = (pd.Series(star[multi]).groupby(plate[multi]).nunique()
              .reindex(plates_all).fillna(0).astype(int))
   good = n_stars.index[n_stars >= min_stars]
   gm = multi & np.isin(plate, good)
   zp_tot = pd.Series(0.0, index=plates_all)
   for _ in range(max_iter):
       ref = pd.Series(mags).groupby([star, bi]).transform("median").to_numpy()
       zp = pd.Series((mags - ref)[gm]).groupby(plate[gm]).median()
       mags -= zp.reindex(plate).fillna(0.0).to_numpy()
       zp_tot = zp_tot.add(zp, fill_value=0.0)
       if float(np.max(np.abs(zp.to_numpy()))) < tol:
           break
   else:
       raise ValueError(f"plate zero-point fit did not converge in {max_iter} iterations")
   ref = pd.Series(mags).groupby([star, bi]).transform("median").to_numpy()
   sig = pd.Series((mags - ref)[gm]).groupby(plate[gm]).apply(
       lambda v: _mad_sigma(v.to_numpy())).reindex(plates_all).fillna(0.0)
   calibrated = (n_stars >= min_stars).to_numpy()
   out = pd.DataFrame({
       "plate": plates_all,
       "band": pd.Series(bi).groupby(plate).first().reindex(plates_all).map(_SSS_BI_BAND).to_numpy(),
       "survey": det["SURVEYNAME"].groupby(plate).first().reindex(plates_all).to_numpy(),
       "n_stars": n_stars.to_numpy(),
       "zeropoint": np.where(calibrated, zp_tot.to_numpy(), 0.0),
       "zeropoint_err": np.where(calibrated,
                                 (sig / np.sqrt(n_stars.clip(lower=1))).to_numpy(), 0.0),
       "calibrated": calibrated,
   })
   if (np.abs(out["zeropoint"]) > 0.5).any():
       raise ValueError("plate zero point exceeds 0.5 mag")
   if not np.isfinite(out[["zeropoint", "zeropoint_err"]].to_numpy()).all():
       raise ValueError("non-finite plate zero point or uncertainty")
   out = out.merge(build_plate_field_surface(det, out, pos), on="plate", how="left")
   if det_limits:
      out = out.merge(build_plate_detection_limits(det, out).drop(columns=["survey"]),
                      on="plate", how="left")
      out["det_model"] = out["det_model"].fillna("none")
   out["model"] = out["model"].fillna("const")
   out[FIELD_COEF] = out[FIELD_COEF].fillna(0.0)
   if (out["field_rms"] > 0.5).any():
       raise ValueError("plate field surface exceeds 0.5 mag rms")
   if out_csv:
      out.to_csv(out_csv, index=False)
   return out


IVEZIC_DAT = _ROOT / "data/stripe82calibStars_v4.2.dat"
_IVEZIC_COLS = {"g": (13, 14, 16), "r": (19, 20, 22), "i": (25, 26, 28)}
_FIELD_REF_BAND = {"SERC-J/EJ": "g", "SERC-R/AAO-R": "r", "SERC-I": "i", "POSSI-E(S)": "r"}
_FIELD_POS = {"const": 0, "plane": 2, "plane_radial": 3}
FIELD_TERMS = ["fx", "fy", "frr", "fxy", "fxmy", "fx3", "fx2y", "fxy2", "fy3"]
FIELD_COEF = ["fx", "fy", "frr", "fmean", "xi_lo", "xi_hi", "eta_lo", "eta_hi"]
FIELD_KFOLD = 5
FIELD_MIN_STARS = 500
FIELD_COL = (-0.5, 3.0)
FIELD_MAG = (14.0, 20.5)


def star_sky_positions() -> np.ndarray:
   with h5py.File(_ROOT / "data/roe_stars.hdf5") as f:
      return np.array([k.split("_") for k in f.keys()], dtype=float)


def ivezic_star_mags() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
   pos = star_sky_positions()
   cols = {c: f"{b}{sx}" for b, t in _IVEZIC_COLS.items()
           for sx, c in zip(("n", "", "sig"), t)}
   cat = pd.read_csv(IVEZIC_DAT, sep=r"\s+", comment="#", header=None,
                     usecols=[1, 2] + list(cols))
   cat.columns = ["ra", "dec"] + [cols[c] for c in sorted(cols)]
   idx, sep, _ = SkyCoord(ra=pos[:, 0] * u.deg, dec=pos[:, 1] * u.deg).match_to_catalog_sky(
      SkyCoord(ra=cat["ra"].to_numpy() * u.deg, dec=cat["dec"].to_numpy() * u.deg))
   ok = (sep.arcsec <= 1.0) & (cat["gn"].to_numpy()[idx] >= 4) & (cat["rn"].to_numpy()[idx] >= 4)
   print(f"Ivezic match: {ok.sum()} of {len(pos)} stars within 1 arcsec "
         f"(median sep {np.median(sep.arcsec[ok]):.3f} arcsec, {len(cat)} catalog rows)")
   pick = lambda c: np.where(ok, cat[c].to_numpy()[idx], np.nan)
   return ({b: pick(b) for b in _IVEZIC_COLS},
           {b: (1.25 * pick(f"{b}sig")) ** 2 for b in _IVEZIC_COLS}, len(pos))


def plate_centres() -> pd.DataFrame:
   with fits.open(_ROOT / "data/survey_table.fits") as h:
      t = h[1].data
      return (pd.DataFrame({"plate": np.asarray(t["PLATEID"]).ravel().astype(np.int64),
                            "cra": np.asarray(t["NOMINALRA"]).ravel().astype(float),
                            "cdec": np.asarray(t["NOMINALDEC"]).ravel().astype(float)})
              .drop_duplicates("plate").set_index("plate"))


def field_coords(plate: np.ndarray, ra: np.ndarray, dec: np.ndarray,
                 ctr: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
   c = plate_centres() if ctr is None else ctr
   xi = ((((ra - c["cra"].reindex(plate).to_numpy()) + 180.0) % 360.0) - 180.0)
   return xi * np.cos(np.deg2rad(dec)), dec - c["cdec"].reindex(plate).to_numpy()


def _field_design(xi: np.ndarray, eta: np.ndarray, k: int) -> np.ndarray:
   return np.column_stack([xi, eta, xi ** 2 + eta ** 2, xi * eta, xi ** 2 - eta ** 2,
                           xi ** 3, xi ** 2 * eta, xi * eta ** 2, eta ** 3])[:, :k]


def _field_nuisance(mag: np.ndarray, col: np.ndarray) -> np.ndarray:
   lo, hi = FIELD_MAG
   m = np.polynomial.legendre.legvander((mag - 0.5 * (lo + hi)) / (0.5 * (hi - lo)), 5)
   return np.column_stack([m, col, col ** 2])


def _field_plate_fit(pl: int, g: pd.DataFrame, rng: np.random.Generator,
                     pos: dict[str, int] = _FIELD_POS) -> dict:
   d = g["d"].to_numpy()
   g = g[np.abs(d - np.median(d)) <= 5.0 * _mad_sigma(d)]
   y = g["d"].to_numpy()
   N = _field_nuisance(g["mag"].to_numpy(), g["col"].to_numpy())
   xi, eta = g["xi"].to_numpy(), g["eta"].to_numpy()
   _, sinv = np.unique(g["star"].to_numpy(), return_inverse=True)
   fold = rng.integers(0, FIELD_KFOLD, sinv.max() + 1)[sinv]
   mse = {}
   for name, k in pos.items():
      A = np.column_stack([N, _field_design(xi, eta, k)])
      e2 = np.empty(len(y))
      for t in range(FIELD_KFOLD):
         tr = fold != t
         e2[~tr] = (y[~tr] - A[~tr] @ np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]) ** 2
      mse[name] = float(e2.mean())
   pick = min(mse, key=mse.get)
   if len(g) < FIELD_MIN_STARS:
      pick = "const"
   k, K = pos[pick], max(pos.values())
   c = np.linalg.lstsq(np.column_stack([N, _field_design(xi, eta, k)]), y,
                       rcond=None)[0][N.shape[1]:]
   c = np.r_[c, np.zeros(K - k)]
   surf = sum(a * x for a, x in zip(c, _field_design(xi, eta, K).T))
   q = lambda v, f: float(np.quantile(v, f))
   return dict(plate=pl, model=pick, n_field=int(len(g)), **dict(zip(FIELD_TERMS, c.astype(float))),
               fmean=float(surf.mean()), xi_lo=q(xi, 0.001), xi_hi=q(xi, 0.999),
               eta_lo=q(eta, 0.001), eta_hi=q(eta, 0.999),
               field_rms=float(np.std(surf)),
               holdout_const=float(np.sqrt(mse["const"])),
               holdout_plane=float(np.sqrt(mse["plane"])),
               holdout_radial=float(np.sqrt(mse["plane_radial"])),
               holdout_pick=float(np.sqrt(mse[pick])))


DET_EDGES = np.arange(17.0, 23.01, 0.25)  # above every native saturation limit
DET_CEN = 0.5 * (DET_EDGES[:-1] + DET_EDGES[1:])
DET_MIN_FOOT = 200
DET_MIN_BINS = 8
DET_KFOLD = 5
DET_MIN_STARS = 500  # footprint standards needed for a spatial depth term
DET_CELL = 0.75  # deg; star-density report cell
# (limit, width) position terms; selection decides where depth structure lives
_DET_POS = {"const": (0, 0), "plane": (2, 0), "plane_radial": (3, 0), "plane_radial_width": (3, 2)}
DET_COEF = ["det_l0", "det_dx", "det_dy", "det_drr", "det_sig", "det_amp", "det_sx", "det_sy",
            "det_lim_lo", "det_lim_hi"]


def _det_curve(m: np.ndarray, lim: float, sig: float, amp: float) -> np.ndarray:
   return amp * norm.sf((np.asarray(m, float) - lim) / sig)


def _det_surface(th: np.ndarray, xi: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   return (th[0] + th[1] * xi + th[2] * eta + th[3] * (xi ** 2 + eta ** 2),
           np.maximum(th[4] + th[6] * xi + th[7] * eta, 0.05))


def _det_obj(th: np.ndarray, m: np.ndarray, xi: np.ndarray, eta: np.ndarray,
             y: np.ndarray) -> tuple[float, np.ndarray]:
   # Bernoulli detection likelihood over footprint standards, as for the zero-point surface
   lim, sg = _det_surface(th, xi, eta)
   t = (lim - m) / sg
   c, ph = norm.cdf(t), norm.pdf(t)
   p = np.clip(th[5] * c, 1e-12, 1.0 - 1e-12)
   nll = -float(np.mean(np.where(y, np.log(p), np.log1p(-p))))
   dp = np.where(y, -1.0 / p, 1.0 / (1.0 - p)) / len(m)
   dl = dp * th[5] * ph / sg
   return nll, np.array([dl.sum(), (dl * xi).sum(), (dl * eta).sum(),
                         (dl * (xi ** 2 + eta ** 2)).sum(), -(dl * t).sum(),
                         (dp * c).sum(), -(dl * t * xi).sum(), -(dl * t * eta).sum()])


def _det_mle(m: np.ndarray, xi: np.ndarray, eta: np.ndarray, y: np.ndarray,
             kl: int, kw: int, x0: np.ndarray) -> np.ndarray:
   free = lambda n, k, b: [b if i < k else (0.0, 0.0) for i in range(n)]
   b = ([(17.0, 24.0)] + free(3, kl, (-5.0, 5.0)) + [(0.05, 2.0), (0.05, 1.0)]
        + free(2, kw, (-1.0, 1.0)))
   return minimize(_det_obj, x0, args=(m, xi, eta, y), jac=True, method="L-BFGS-B",
                   bounds=b, options=dict(maxiter=500)).x


def _det_binned(m: np.ndarray, y: np.ndarray, pred: np.ndarray,
                cell: np.ndarray | None = None) -> float:
   # detected vs predicted fraction per mag bin; per spatial cell it sees depth gradients
   k = np.searchsorted(DET_EDGES, m, side="right") - 1
   k = k if cell is None else k + len(DET_CEN) * cell
   nf = np.bincount(k, minlength=len(DET_CEN)).astype(float)
   ok = nf >= (DET_MIN_FOOT if cell is None else 50)
   if int(ok.sum()) < DET_MIN_BINS:
      return np.nan
   h = lambda w: np.bincount(k, weights=w, minlength=len(nf))[ok] / nf[ok]
   return float(np.sqrt(np.mean((h(y.astype(float)) - h(pred)) ** 2)))


def _det_plate_fit(pl: int, sv: str, m: np.ndarray, xi: np.ndarray, eta: np.ndarray,
                   y: np.ndarray, fold: np.ndarray) -> dict:
   cid = np.unique(np.floor(np.column_stack([xi, eta]) / DET_CELL).astype(np.int64), axis=0,
                   return_inverse=True, return_counts=True)
   ci, ncell = cid[1].ravel(), cid[2]
   nf = np.histogram(m, bins=DET_EDGES)[0].astype(float)
   fr = np.divide(np.histogram(m[y], bins=DET_EDGES)[0], nf, out=np.zeros_like(nf), where=nf > 0)
   a0 = float(np.clip(np.max(fr), 0.5, 1.0))
   x0 = np.array([float(np.clip(DET_CEN[np.argmin(np.abs(fr - 0.5 * a0))], 17.1, 23.9)),
                  0.0, 0.0, 0.0, 0.5, min(a0, 0.999), 0.0, 0.0])
   cv, oof, th = {}, {}, {}
   for name, (kl, kw) in _DET_POS.items():
      e, pr = 0.0, np.empty(len(m))
      for t in range(DET_KFOLD):
         tr = fold != t
         q = _det_mle(m[tr], xi[tr], eta[tr], y[tr], kl, kw, x0)
         e += _det_obj(q, m[~tr], xi[~tr], eta[~tr], y[~tr])[0] * int((~tr).sum())
         pr[~tr] = _det_curve(m[~tr], *_det_surface(q, xi[~tr], eta[~tr]), q[5])
      cv[name], oof[name] = e / len(m), pr
      th[name] = _det_mle(m, xi, eta, y, kl, kw, x0)
      x0 = th[name].copy()
   pick = min(cv, key=cv.get)
   if len(m) < DET_MIN_STARS:
      pick = "const"
   q, qc = th[pick], th["const"]
   lim, sg = _det_surface(q, xi, eta)
   return dict(plate=int(pl), survey=sv, det_model=pick, det_n_foot=int(len(m)),
               det_n_cell_min=int(ncell.min()), det_n_cell=int(len(ncell)),
               det_n_cell_lt50=int((ncell < 50).sum()),
               det_limit=float(qc[0]), det_sigma=float(qc[4]), det_plateau=float(q[5]),
               **dict(zip(DET_COEF[:8], [float(v) for v in q[:8]])),
               det_lim_lo=float(np.quantile(lim, 0.001)), det_lim_hi=float(np.quantile(lim, 0.999)),
               det_lim_rms=float(np.std(lim)),
               det_lim_span=float(np.ptp(np.quantile(lim, [0.05, 0.95]))),
               det_rms=_det_binned(m, y, oof[pick]),
               det_rms_const=_det_binned(m, y, oof["const"]),
               det_cell_rms=_det_binned(m, y, oof[pick], ci),
               det_cell_rms_const=_det_binned(m, y, oof["const"], ci),
               **{f"det_cv_{n}": float(cv[n]) for n in _DET_POS})


def build_plate_detection_limits(det: pd.DataFrame, zp: pd.DataFrame) -> pd.DataFrame:
   # per-plate detection probability from footprint standards (offset curve cannot split limit from cubic);
   # plane + radial depth surface vs constant, five-fold CV held out by star
   mags, _, npos = ivezic_star_mags()
   pos = star_sky_positions()
   z = zp.set_index("plate")
   fold5 = np.random.default_rng(23).integers(0, DET_KFOLD, npos)
   rows = []
   for pl, g in det.groupby("plate"):
      if pl not in z.index:
         continue
      r = z.loc[pl]
      sv = str(r["survey"]).strip()
      ref = mags.get(_FIELD_REF_BAND.get(sv))
      if ref is None or not np.isfinite(r["xi_lo"]):
         continue
      xi, eta = field_coords(np.full(npos, pl, dtype=np.int64), pos[:, 0], pos[:, 1])
      inside = ((xi >= r["xi_lo"]) & (xi <= r["xi_hi"]) & (eta >= r["eta_lo"])
                & (eta <= r["eta_hi"]) & np.isfinite(ref)
                & (ref >= DET_EDGES[0]) & (ref < DET_EDGES[-1]))
      seen = np.zeros(npos, dtype=bool)
      seen[np.unique(g["star"].to_numpy())] = True
      k = np.flatnonzero(inside)
      if len(k) < DET_MIN_FOOT * DET_MIN_BINS:
         continue
      rows.append(_det_plate_fit(pl, sv, ref[k], xi[k], eta[k], seen[k], fold5[k]))
   out = pd.DataFrame(rows)
   if out.empty:
      raise ValueError("no plate detection curve converged")
   if not out["det_plateau"].between(0.8, 1.0).all():
      raise ValueError("plate detection plateau outside [0.8, 1.05]")
   print(f"plate detection curves: {len(out)} plates, limit "
         f"{out['det_limit'].min():.2f}-{out['det_limit'].max():.2f}, sigma "
         f"{out['det_sigma'].min():.2f}-{out['det_sigma'].max():.2f}; spatial model "
         + ", ".join(f"{k} {v}" for k, v in out["det_model"].value_counts().items()))
   for sv, g in out.groupby("survey"):
      print(f"  {sv}: {len(g)} plates, footprint standards {int(g['det_n_foot'].min())}-"
            f"{int(g['det_n_foot'].max())}, min stars per {DET_CELL:.2f} deg cell "
            f"{int(g['det_n_cell_min'].min())} (median {int(g['det_n_cell_min'].median())}, "
            f"{int(g['det_n_cell'].median())} cells, {int(g['det_n_cell_lt50'].median())} "
            f"of them under 50), limit rms across the plate "
            f"{g['det_lim_rms'].median():.3f} mag, 5-95 pct span {g['det_lim_span'].median():.3f}; "
            f"held-out detected-fraction rms {g['det_rms_const'].median():.4f} -> "
            f"{g['det_rms'].median():.4f} in magnitude bins and "
            f"{g['det_cell_rms_const'].median():.4f} -> {g['det_cell_rms'].median():.4f} "
            f"in magnitude bins per cell; held-out log-loss "
            f"{g['det_cv_const'].median():.5f} const, {g['det_cv_plane'].median():.5f} plane, "
            f"{g['det_cv_plane_radial'].median():.5f} plane+radial, "
            f"{g['det_cv_plane_radial_width'].median():.5f} plane+radial+width; "
            f"{int((g['det_cv_plane_radial_width'] < g['det_cv_plane_radial']).sum())} plates "
            f"prefer a spatial width")
   return out


def build_plate_field_surface(det: pd.DataFrame, zp: pd.DataFrame,
                              pos: dict[str, int] | None = None) -> pd.DataFrame:
   # de Vries 2005: local calibration; smooth per-plate surface vs Ivezic CCD, held-out mse
   mags, _, npos = ivezic_star_mags()
   st = det["star"].to_numpy()
   if st.max() >= npos:
      raise ValueError("star index exceeds hdf5 key count")
   sv = det["SURVEYNAME"].astype(str).str.strip().to_numpy()
   ref = np.full(len(det), np.nan)
   for name, b in _FIELD_REF_BAND.items():
      m = sv == name
      ref[m] = mags[b][st[m]]
   plate = det["plate"].to_numpy().astype(np.int64)
   mag = (det["SMAG"].to_numpy(float)
          - zp.set_index("plate")["zeropoint"].reindex(plate).fillna(0.0).to_numpy())
   xi, eta = field_coords(plate, det["ra"].to_numpy(float), det["dec"].to_numpy(float))
   col = (mags["g"] - mags["r"])[st]
   f = pd.DataFrame(dict(plate=plate, star=st, d=mag - ref, mag=mag, col=col, xi=xi, eta=eta))
   f = f[np.isfinite(f[["d", "col", "xi", "eta"]].to_numpy()).all(1)
         & f["col"].between(*FIELD_COL) & f["mag"].between(*FIELD_MAG)
         & f["plate"].isin(zp.loc[zp["calibrated"], "plate"])]
   rng = np.random.default_rng(19)
   return pd.DataFrame([_field_plate_fit(int(pl), g, rng, pos or _FIELD_POS) for pl, g in f.groupby("plate")])


def plate_det_limit(zp: pd.DataFrame, plate: np.ndarray, ra: np.ndarray,
                    dec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   # outside fitted footprint: plate's single limit, no gradient extrapolation
   z = zp.set_index("plate")
   plate = np.asarray(plate, dtype=np.int64)
   g = lambda c: z[c].reindex(plate).to_numpy(float)
   xi, eta = field_coords(plate, np.asarray(ra, float), np.asarray(dec, float))
   lim = np.clip(g("det_l0") + g("det_dx") * xi + g("det_dy") * eta
                 + g("det_drr") * (xi ** 2 + eta ** 2), g("det_lim_lo"), g("det_lim_hi"))
   sig = np.maximum(g("det_sig") + g("det_sx") * xi + g("det_sy") * eta, 0.05)
   inside = ((xi >= g("xi_lo")) & (xi <= g("xi_hi")) & (eta >= g("eta_lo"))
             & (eta <= g("eta_hi")) & np.isfinite(lim) & np.isfinite(sig))
   return (np.where(inside, lim, g("det_limit")), np.where(inside, sig, g("det_sigma")), inside)


def plate_zp_offset(zp: pd.DataFrame, plate: np.ndarray, ra: np.ndarray,
                    dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   # single evaluation point for stars and quasars
   z = zp.set_index("plate")
   plate = np.asarray(plate, dtype=np.int64)
   g = lambda c: z[c].reindex(plate).fillna(0.0).to_numpy(float)
   xi, eta = field_coords(plate, np.asarray(ra, float), np.asarray(dec, float))
   terms = [c for c in FIELD_TERMS if c in z]
   fld = sum(g(c) * x for c, x in zip(terms, _field_design(xi, eta, len(terms)).T)) - g("fmean")
   inside = ((xi >= g("xi_lo")) & (xi <= g("xi_hi"))
             & (eta >= g("eta_lo")) & (eta <= g("eta_hi")) & np.isfinite(fld))
   return g("zeropoint") + np.where(inside, fld, 0.0), g("zeropoint_err")


def build_calibration_chain(max_iter: int = 4, tol: float = 0.002) -> tuple[pd.DataFrame, pd.DataFrame]:
   # R1 map and plate surfaces coupled: solved alternately until the map converges
   calib_csv = str(_ROOT / "data/sss_plate_calibration.csv")
   zp_csv = str(_ROOT / "data/sss_plate_zeropoints.csv")
   calib = build_sss_calibration(calib_csv)
   zp = build_plate_zeropoints(star_zp_detections(calib), zp_csv)
   for i in range(1, max_iter + 1):
      prev = calib[calib["band"] == "R1"]
      calib = build_sss_calibration(calib_csv, zp)
      zp = build_plate_zeropoints(star_zp_detections(calib), zp_csv)
      cur = calib[calib["band"] == "R1"]
      step = float(np.abs(np.interp(cur["mag_center"], prev["mag_center"], prev["offset"])
                          - cur["offset"]).max())
      print(f"R1 map iteration {i}: max offset change {step:.4f} mag")
      if step < tol:
         break
   else:
      raise ValueError(f"R1 map did not converge in {max_iter} iterations")
   # zero points fitted on the on-disk calibration downstream reads
   calib = pd.read_csv(calib_csv)
   zp = build_plate_zeropoints(star_zp_detections(calib), zp_csv)
   return calib, zp


def apply_plate_zeropoints(df_sss: pd.DataFrame, zp: pd.DataFrame) -> pd.DataFrame:
   df = df_sss.copy()
   pid = df["PLATEID"].to_numpy().astype(np.int64)
   matched = np.isin(pid, zp.loc[zp["calibrated"], "plate"].to_numpy())
   if matched.mean() < 0.95:
       raise ValueError(f"only {matched.mean():.1%} of SSS epochs on a calibrated plate")
   col = lambda n: df[n if n in df.columns else n.lower()].to_numpy(float)
   off, err = plate_zp_offset(zp, pid, col("RA"), col("DEC"))
   df["SMAG"] = df["SMAG"].to_numpy(dtype=float) - off
   for c in ["SMAG_ERR", "SMAG_ERR_CAT"]:
       if c in df.columns:
           df[c] = np.sqrt(df[c].to_numpy(dtype=float) ** 2 + err ** 2)
   return df


TRUNC_CSV = _ROOT / "data/plate_truncation_correction.csv"
TRUNC_PARAMS = ["a0", "a1", "a2", "a3", "mag_limit", "sigma"]
_NATIVE_SDSS = {"B_J": "g", "R": "r", "I": "i"}


def truncation_model(m: np.ndarray, a0: float, a1: float, a2: float, a3: float,
                     lim: float, sig: float) -> np.ndarray:
   # cubic nonlinearity + truncated-Gaussian mean at the plate limit
   x = np.asarray(m, dtype=float) - 20.0
   u = (lim - np.asarray(m, dtype=float)) / sig
   return a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 - sig * np.exp(norm.logpdf(u) - norm.logcdf(u))


def truncation_curve() -> pd.DataFrame:
   if not TRUNC_CSV.exists():
       raise FileNotFoundError(f"{TRUNC_CSV} missing; run "
                               "star_null_structure_function.py --plate-ccd first")
   return pd.read_csv(TRUNC_CSV)


def truncation_offset(surveyname, ref_mag, curve: pd.DataFrame) -> np.ndarray:
   # on CCD reference mag (untruncated); flat outside fit range
   sn = pd.Series(surveyname).astype(str).str.strip().to_numpy()
   ref = np.asarray(ref_mag, dtype=float)
   off = np.zeros(len(ref))
   for r in curve.itertuples():
       m = (sn == r.survey) & np.isfinite(ref)
       if m.any():
           off[m] = truncation_model(np.clip(ref[m], r.fit_lo, r.fit_hi),
                                     *[getattr(r, c) for c in TRUNC_PARAMS])
   return off


TIE_CSV = _ROOT / "data/sss_plate_ccd_tie.csv"
TIE_MAG = (17.0, 21.0)


def plate_ccd_tie(plate, ref_mag) -> np.ndarray:
   # ties relative zero points to SDSS per plate
   t = pd.read_csv(TIE_CSV).set_index("plate")
   p = np.asarray(plate, dtype=np.int64)
   a, b = (t[c].reindex(p).fillna(0.0).to_numpy(float) for c in ("tie_a", "tie_b"))
   m = np.clip(np.asarray(ref_mag, dtype=float), *TIE_MAG) - 19.0
   return np.where(np.isfinite(m), a + b * m, 0.0)


# lowest u = (limit - reference)/sigma validated on standards
# (10-20 pct to u = -2, star_null --plate-ccd); V held flat below
TRUNC_U_MIN = -2.0


def truncation_variance(surveyname, plate, ra, dec, ref_mag, curve: pd.DataFrame,
                        zp: pd.DataFrame) -> np.ndarray:
   # survivors carry V x untruncated variance; pair estimator divides it out;
   # V from the epoch's own detection surface, else survey curve, else uncorrected
   sn = pd.Series(surveyname).astype(str).str.strip().to_numpy()
   c = curve.set_index("survey")
   lim, sig = (np.where(np.isfinite(a), a, c[b].reindex(sn).to_numpy(float))
               for a, b in zip(plate_det_limit(zp, plate, ra, dec)[:2], ("mag_limit", "sigma")))
   u = np.maximum((lim - np.clip(np.asarray(ref_mag, float), DET_EDGES[0], DET_EDGES[-1])) / sig,
                  TRUNC_U_MIN)
   with np.errstate(invalid="ignore"):
       lam = np.exp(norm.logpdf(u) - norm.logcdf(u))
       v = np.clip(1.0 - u * lam - lam ** 2, 1e-3, 1.0)
   return np.where(np.isfinite(v), v, 1.0)


def ccd_median_mags(df_base: pd.DataFrame) -> pd.Series:
   return df_base.groupby(["OBJID", "band"])["mag"].median().rename("modmed")


def apply_truncation_correction(df_sss: pd.DataFrame, modmed: pd.Series,
                                zp: pd.DataFrame) -> pd.DataFrame:
   df = df_sss.copy()
   band = df["BAND_NATIVE"].astype(str).map(_NATIVE_SDSS)
   ref = modmed.reindex(pd.MultiIndex.from_arrays([df["OBJID"].astype(str), band])).to_numpy()
   col = lambda n: df[n if n in df.columns else n.lower()].to_numpy(float)
   curve = truncation_curve()
   off = truncation_offset(df["SURVEYNAME"], ref, curve)
   have = np.isfinite(ref)
   sn = df["SURVEYNAME"].astype(str).str.strip().to_numpy()
   lo = pd.Series(curve.set_index("survey")["fit_lo"]).reindex(sn).to_numpy(float)
   hi = pd.Series(curve.set_index("survey")["fit_hi"]).reindex(sn).to_numpy(float)
   # no constraint past fitted CCD range; SERC-I has no curve
   df["calib_ok"] = (df["calib_ok"].to_numpy() & have
                     & (np.isnan(lo) | ((ref >= lo) & (ref <= hi))))
   for s in sorted(set(df["SURVEYNAME"].astype(str).str.strip())):
       m = (df["SURVEYNAME"].astype(str).str.strip() == s).to_numpy()
       if not m.any():
           continue
       print(f"truncation correction {s}: {int(have[m].sum())}/{int(m.sum())} epochs with a CCD "
             f"reference, median {np.median(off[m]):+.4f} mag, "
             f"5/95 pct {np.percentile(off[m], 5):+.4f}/{np.percentile(off[m], 95):+.4f}, "
             f"|shift| > 0.05 for {int((np.abs(off[m]) > 0.05).sum())}")
   df["SMAG"] = df["SMAG"].to_numpy(dtype=float) - off - plate_ccd_tie(df["PLATEID"], ref)
   df["trunc_v"] = truncation_variance(df["SURVEYNAME"], df["PLATEID"], col("RA"), col("DEC"),
                                       ref, curve, zp)
   lam0 = float(np.exp(norm.logpdf(TRUNC_U_MIN) - norm.logcdf(TRUNC_U_MIN)))
   vfl = 1.0 - TRUNC_U_MIN * lam0 - lam0 ** 2
   ins = plate_det_limit(zp, df["PLATEID"], col("RA"), col("DEC"))[2]
   for s in sorted(set(df["SURVEYNAME"].astype(str).str.strip())):
       m = (df["SURVEYNAME"].astype(str).str.strip() == s).to_numpy()
       v = df["trunc_v"].to_numpy()[m]
       print(f"truncation variance {s}: median V {np.median(v):.3f}, "
             f"5/95 pct {np.percentile(v, 5):.3f}/{np.percentile(v, 95):.3f}; "
             f"{np.mean(np.abs(v - vfl) < 1e-9):.4f} on the u = {TRUNC_U_MIN} floor, "
             f"{1.0 - np.mean(ins[m]):.4f} outside the fitted footprint")
   return df


CCD_TRUNC_CSV = _ROOT / "data/ccd_truncation_correction.csv"
CCD_BINS = np.arange(17.0, 23.51, 0.25)
CCD_MIN_OBJ = 60
CCD_BRIGHT = 19.5
CCD_BANDS = ["g", "r", "i", "z"]
CCD_MIN_FRAC = 0.2
# held-out test unbiased to here; beyond, g over-corrects (+0.072 mag at 22.6, +0.120 at 23.1)
CCD_VALID_MAX = 22.4


def ccd_reference_mags(df_base: pd.DataFrame) -> pd.Series:
   # qsogen catalog mags: deep, no epoch detection selection; SDSS median fills gaps
   cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet",
                         columns=["objectId"] + [f"sdss_{b}_qg" for b in CCD_BANDS])
   cat["OBJID"] = cat["objectId"].astype(str)
   qg = cat.set_index("OBJID")[[f"sdss_{b}_qg" for b in CCD_BANDS]]
   qg.columns = CCD_BANDS
   qg = qg.stack().rename("ref")
   qg.index = qg.index.set_names(["OBJID", "band"])
   sd = (df_base[df_base["survey"] == "sdss"].groupby(["OBJID", "band"])["mag"].median()
         .rename("ref"))
   z0 = float(np.nanmedian((sd - qg).dropna()[qg.reindex(sd.index) < CCD_BRIGHT]))
   return (qg + z0).combine_first(sd)


def build_ccd_truncation(df_base: pd.DataFrame, ref: pd.Series | None = None,
                        out_csv: Path | None = CCD_TRUNC_CSV) -> pd.DataFrame:
   ref = ccd_reference_mags(df_base) if ref is None else ref
   med = df_base.groupby(["OBJID", "band", "survey"])["mag"].median().rename("med")
   n = df_base.groupby(["OBJID", "band", "survey"]).size().rename("n")
   d = pd.concat([med, n], axis=1).reset_index()
   d["ref"] = ref.reindex(pd.MultiIndex.from_arrays([d["OBJID"], d["band"]])).to_numpy()
   d = d[np.isfinite(d["ref"]) & d["band"].isin(CCD_BANDS)]
   d["node"] = np.clip(np.digitize(d["ref"], CCD_BINS) - 1, 0, len(CCD_BINS) - 2)
   rows = []
   for (sv, b), g in d.groupby(["survey", "band"], sort=True):
       if len(g) < 10 * CCD_MIN_OBJ:
           continue
       z0 = float(np.median((g["med"] - g["ref"])[g["ref"] < CCD_BRIGHT]))
       t = g.assign(o=g["med"] - g["ref"] - z0).groupby("node").agg(
           o=("o", "median"), k=("n", "mean"), N=("o", "size"))
       t = t[t["N"] >= CCD_MIN_OBJ]
       x = CCD_BINS[t.index.to_numpy()] + 0.125
       # truncation offset one-sided and monotone: PAVA
       o = np.minimum(_pava_decreasing(t["o"].to_numpy(), t["N"].to_numpy()), 0.0)
       f = t["k"].to_numpy() / max(t["k"].to_numpy()[x < CCD_BRIGHT].max(), 1e-9)
       lim, sig = _ccd_det_fit(x, f, t["N"].to_numpy())
       u = np.maximum((lim - x) / sig, TRUNC_U_MIN)
       lam = np.exp(norm.logpdf(u) - norm.logcdf(u))
       # k: source share of selection variance (from offset) = share of second moment lost
       k = np.clip(np.where(lam > 1e-6, -o / (sig * np.maximum(lam, 1e-6)), 0.0), 0.0, 1.0)
       v = np.clip(1.0 - k * (u * lam + lam ** 2), 1e-2, 1.0)
       rows.append(pd.DataFrame(dict(survey=sv, band=b, ref_mag=x, n_objects=t["N"].to_numpy(),
                                     zero=z0, offset_mag=o, det_frac=np.minimum(f, 1.0),
                                     det_limit=lim, det_sigma=sig, k_source=k, retained_v=v)))
   out = pd.concat(rows, ignore_index=True)
   # rewrite only on change, keeps mtime behind consumers
   if out_csv is not None:
       txt = out.to_csv(index=False)
       if not out_csv.exists() or out_csv.read_text() != txt:
           out_csv.write_text(txt)
   return out


def _pava_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
   # truncation offset only deepens toward the limit
   out: list[list[float]] = []
   for a, m in zip(np.asarray(y, float), np.asarray(w, float)):
       out.append([a, m, 1])
       while len(out) > 1 and out[-2][0] < out[-1][0]:
           a2, w2, n2 = out.pop()
           a1, w1, n1 = out.pop()
           out.append([(a1 * w1 + a2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
   return np.repeat([r[0] for r in out], [int(r[2]) for r in out])


def _ccd_det_fit(x: np.ndarray, f: np.ndarray, w: np.ndarray) -> tuple[float, float]:
   p, _ = curve_fit(lambda m, lim, sig: norm.cdf((lim - m) / sig), x, np.clip(f, 0.0, 1.0),
                    p0=[x[np.argmin(np.abs(f - 0.5))], 0.8], sigma=1.0 / np.sqrt(w),
                    bounds=([15.0, 0.05], [26.0, 5.0]), maxfev=50000)
   return float(p[0]), float(p[1])


def ccd_truncation_curve() -> pd.DataFrame:
   if not CCD_TRUNC_CSV.exists():
       raise FileNotFoundError(f"{CCD_TRUNC_CSV} missing; rebuilt by assemble_lightcurves.main")
   return pd.read_csv(CCD_TRUNC_CSV)


def apply_ccd_truncation(df_base: pd.DataFrame, curve: pd.DataFrame,
                        ref: pd.Series | None = None) -> pd.DataFrame:
   df = df_base.copy()
   ref = ccd_reference_mags(df) if ref is None else ref
   r = ref.reindex(pd.MultiIndex.from_arrays([df["OBJID"].astype(str), df["band"]])).to_numpy()
   off, v, fr = np.zeros(len(df)), np.ones(len(df)), np.ones(len(df))
   ok = np.isfinite(r)
   has = np.zeros(len(df), dtype=bool)
   sv, bd = df["survey"].to_numpy(), df["band"].to_numpy()
   for (s, b), g in curve.groupby(["survey", "band"], sort=True):
       m = ok & (sv == s) & (bd == b)
       if not m.any():
           continue
       has[m] = True
       # flat outside measured reference range
       off[m] = np.interp(r[m], g["ref_mag"], g["offset_mag"])
       v[m] = np.interp(r[m], g["ref_mag"], g["retained_v"])
       fr[m] = np.interp(r[m], g["ref_mag"], g["det_frac"])
       print(f"ccd truncation {s} {b}: {int(m.sum())} epochs, median offset "
             f"{np.median(off[m]):+.4f}, 5/95 pct {np.percentile(off[m], 5):+.4f}/"
             f"{np.percentile(off[m], 95):+.4f}, median V {np.median(v[m]):.3f}, "
             f"{float(np.mean(off[m] < -0.05)):.4f} beyond 0.05 mag, "
             f"{float(np.mean(fr[m] < 0.5)):.4f} below half completeness")
   df["mag"] = df["mag"].to_numpy(float) - off
   df["trunc_v"] = v
   # flagged below CCD_MIN_FRAC or past CCD_VALID_MAX; calibok drops them
   # u has no curve; flag covers corrected (survey, band) only
   df["calib_ok"] = ok & (fr >= CCD_MIN_FRAC) & (~has | (r <= CCD_VALID_MAX))
   for b in sorted(set(df["band"])):
       m = (bd == b)
       print(f"ccd validity {b}: {int((m & ~df['calib_ok'].to_numpy()).sum())} of {int(m.sum())} "
             f"epochs flagged, {int((m & has & (r > CCD_VALID_MAX)).sum())} beyond ref "
             f"{CCD_VALID_MAX}")
   return df


def match_groups_to_catalog(df_groups: pd.DataFrame, catalog_parquet: str, max_sep_arcsec: float = 1.0) -> pd.DataFrame:
   g = (
       df_groups[["GROUP_ID", "RA", "DEC"]]
       .dropna()
       .groupby("GROUP_ID", as_index=False)
       .median()
   )
   cat = pd.read_parquet(catalog_parquet, columns=["objectId", "RA", "DEC", "Z_DR16Q"])
   coords_cat = SkyCoord(ra=cat["RA"].to_numpy() * u.deg, dec=cat["DEC"].to_numpy() * u.deg, frame="icrs")
   coords_grp = SkyCoord(ra=g["RA"].to_numpy() * u.deg, dec=g["DEC"].to_numpy() * u.deg, frame="icrs")
   idx, sep2d, _ = coords_grp.match_to_catalog_sky(coords_cat)
   sep = sep2d.to(u.arcsec).value
   ok = sep <= float(max_sep_arcsec)
   return (
       pd.DataFrame({
           "GROUP_ID": g.loc[ok, "GROUP_ID"].to_numpy(),
           "OBJID": cat["objectId"].to_numpy()[idx[ok]].astype(str),
           "z": cat["Z_DR16Q"].to_numpy()[idx[ok]].astype(float),
           "sep_arcsec": sep[ok],
       })
       .sort_values("sep_arcsec")
       .drop_duplicates(subset=["GROUP_ID"], keep="first")
       .drop(columns=["sep_arcsec"])
       .reset_index(drop=True)
   )


def apply_mag_guard(df_sss: pd.DataFrame, df_base: pd.DataFrame, max_dm: float) -> pd.DataFrame:
   # |plate - CCD| after per-(survey, band) median offset
   modmed = ccd_median_mags(df_base)
   df = df_sss.merge(modmed, on=["OBJID", "band"], how="left")
   dm = df["mag"] - df["modmed"]
   off = dm.groupby([df["survey"], df["band"]]).transform("median")
   keep = (dm - off).abs() < max_dm
   keep |= df["modmed"].isna()
   print(f"mag guard {max_dm}: keep {int(keep.sum())} of {len(df)} SSS epochs")
   return df.loc[keep].drop(columns=["modmed"]).reset_index(drop=True)


def build_sss_epochs(radius: float = 1.0, sat_cut: bool = True,
                     df_base: pd.DataFrame | None = None) -> pd.DataFrame:
   if (_ROOT / "data/plate_native_calibration.json").exists():
       from plate_magnitude_models import native_longform
       return native_longform(radius, sat_cut)
   df_sss_raw = sss_raw()
   if len(df_sss_raw) == 0:
       raise ValueError("no SSS rows loaded")

   calib_csv = _ROOT / "data/sss_plate_calibration.csv"
   zp_csv = _ROOT / "data/sss_plate_zeropoints.csv"
   if calib_csv.exists() and zp_csv.exists():
       calib, zp = pd.read_csv(calib_csv), pd.read_csv(zp_csv)
   else:
       calib, zp = build_calibration_chain()

   df_grp = assign_group_ids_by_sky(df_sss_raw, radius_arcsec=radius)
   gid_map = match_groups_to_catalog(df_grp, str(_ROOT / "data/S82/Catalog.parquet"),
                                     max_sep_arcsec=radius)
   df_sss_m = df_grp.drop(columns=["OBJID", "z"], errors="ignore").merge(gid_map, on="GROUP_ID", how="inner")
   n_match = int((df_sss_m["SURVEYNAME"].astype(str) == "POSSI-E(S)").sum())
   df_sss_m = drop_blended_parents(df_sss_m)
   possi_m = (df_sss_m["SURVEYNAME"].astype(str) == "POSSI-E(S)").to_numpy()
   df_sss_m = apply_sss_calibration(df_sss_m, calib)
   df_sss_m = apply_plate_zeropoints(df_sss_m, zp)
   if df_base is not None:
       df_sss_m = apply_truncation_correction(df_sss_m, ccd_median_mags(df_base), zp)
   n_sat = int((possi_m & (df_sss_m["SMAG"].to_numpy(float) < SATURATION_LIMITS["R1"])).sum()) if sat_cut else 0
   df_sss = sss_to_sdss_longform(df_sss_m, sat_cut=sat_cut)
   if len(df_sss) == 0:
       raise ValueError("no SSS epochs after conversion")
   possi_raw = df_sss_raw["SURVEYNAME"].astype(str) == "POSSI-E(S)"
   r1lo = calib.loc[calib["band"] == "R1", "mag_center"].min()
   n_kept = int((df_sss["survey"] == "sss_possi").sum())
   print(f"POSS-I ledger: archive {int(possi_raw.sum())} detections "
         f"({int((df_sss_raw.loc[possi_raw, 'SMAG'] < r1lo).sum())} brightward of the first R1 node), "
         f"outside the {radius} arcsec match {int(possi_raw.sum()) - n_match}, blend rule "
         f"{n_match - int(possi_m.sum())}, saturation {n_sat}, template range "
         f"{int(possi_m.sum()) - n_sat - n_kept}, kept {n_kept}")
   return df_sss


def main(radius: float = 1.0, out: Path | None = None, mag_guard: float | None = None) -> None:
   df_base = concat_light_curves_df(data_dir=str(_ROOT / "data/S82")).rename(columns={"objectId": "OBJID"})
   if df_base["OBJID"].isna().any():
       raise ValueError("NaN OBJID rows after concat")
   df_base["OBJID"] = df_base["OBJID"].astype(str)
   # CCD surveys detection limited (ZTF first): same treatment as plates before medians
   df_base = apply_ccd_truncation(df_base, build_ccd_truncation(df_base))

   df_sss = build_sss_epochs(radius=radius, df_base=df_base).drop(columns=["smag"])
   if mag_guard is not None:
       df_sss = apply_mag_guard(df_sss, df_base, mag_guard)

   df_total = assemble_total_lightcurve(df_base, df_sss)

   n_possi = int((df_total["survey"] == "sss_possi").sum())
   native_path = _ROOT / "data/plate_native_quasar_epochs.parquet"
   if native_path.exists():
       native = pd.read_parquet(native_path, columns=["SURVEYNAME", "supported"])
       if n_possi != int((native.SURVEYNAME.eq("POSSI-E(S)") & native.supported).sum()):
           raise ValueError("Native POSS-I epoch count differs from its support ledger")
   elif not (5000 <= n_possi <= 15000):
       raise ValueError(f"POSSI-E(S) epoch count {n_possi} outside [5000, 15000]")
   is_sss = df_total["survey"].str.startswith("sss")
   n_both = len(np.intersect1d(df_total.loc[is_sss, "OBJID"].unique(),
                               df_total.loc[~is_sss, "OBJID"].unique()))
   if n_both <= 1000:
       raise ValueError(f"only {n_both} objects with both historical and modern epochs")

   flag = ~df_total["calib_ok"].to_numpy(bool)
   pos = (df_total["survey"] == "sss_possi").to_numpy()
   print(f"calib_ok False: {int((flag & pos).sum())} of {int(pos.sum())} POSS-I epochs "
         f"({100 * (flag & pos).sum() / pos.sum():.1f} percent), {int((flag & is_sss).sum())} of "
         f"{int(is_sss.sum())} plate epochs")
   print("plate epochs by survey/band: "
         + " ".join(f"{s}/{b} {n}" for (s, b), n in df_total[is_sss].groupby(["survey", "band"]).size().items())
         + f" total {int(is_sss.sum())}")
   out = _ROOT / "data/S82/total_lightcurves.parquet" if out is None else Path(out)
   df_total.to_parquet(out, index=False)
   print(f"rows {len(df_total)} objects {df_total['OBJID'].nunique()} sss {int(is_sss.sum())} "
         f"possi {n_possi} both {n_both} -> {out}")


if __name__ == "__main__":
   import argparse
   argparse.ArgumentParser().parse_args()
   main()
