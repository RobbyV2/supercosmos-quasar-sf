from pathlib import Path
import tarfile

import h5py
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.stats import norm
import astropy.units as u

_ROOT = Path(__file__).resolve().parents[1]

def read_bandpass(fname):
   tbl_bp = Table.read(str(_ROOT / "data" / fname), format='ascii')
   wcol, tcol = tbl_bp.colnames[:2]
   w = np.asarray(tbl_bp[wcol], dtype=float)
   T = np.asarray(tbl_bp[tcol], dtype=float)
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


def vac_to_air(w):
   s2 = (1e4 / w) ** 2
   return w / (1 + 8.34254e-5 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))


def ab_minus_vega(wf, tf):
   # CALSPEC vacuum wavelengths -> air, as the passbands
   d = fits.getdata(_ROOT / "data/alpha_lyr_stis_012.fits", 1)
   return abmag(vac_to_air(np.asarray(d["WAVELENGTH"], dtype=float)), np.asarray(d["FLUX"], dtype=float),
                np.asarray(wf, dtype=float), np.asarray(tf, dtype=float))


filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}
bands_sdss = ["u", "g", "r", "i", "z"]


def sdss_epochs() -> tuple[pd.Index, pd.DataFrame]:
   with tarfile.open(_ROOT / "data/S82/macleod2012_s82_lightcurves.tar.gz") as t:
       files = [(m.name.rsplit("/", 1)[1], t.extractfile(m).read()) for m in t if m.isfile()]
   a = np.array(b"".join(f for _, f in files).split(), dtype=float).reshape(-1, 17)
   dbid = np.repeat([n for n, _ in files], [f.count(b"\n") for _, f in files])
   _, first = np.unique(dbid, return_index=True)
   ids = match_groups_to_catalog(pd.DataFrame({"GROUP_ID": dbid[first], "RA": a[first, 15] % 360, "DEC": a[first, 16]}),
                                 _ROOT / "data/S82/Catalog.parquet").set_index("GROUP_ID").OBJID
   lc = pd.concat([pd.DataFrame({"objectId": ids.reindex(dbid).to_numpy(), "band": b, "time": a[:, 3 * k],
                                 "mag": a[:, 3 * k + 1], "magerr": a[:, 3 * k + 2]}) for k, b in enumerate(bands_sdss)])
   ok = lc.objectId.notna() & (lc.mag > 0) & (lc.mag < 90) & (lc.magerr < 99) & (lc.time > 1)
   return pd.Index(ids), lc[ok].reset_index(drop=True)


def concat_light_curves_df(
   filter_object_ids=None,
   skip=None,
   N=None,
):
   data_dir = _ROOT / "data/S82"
   cat = pd.read_parquet(data_dir / "Catalog.parquet").set_index("idx")

   ps1 = pd.read_parquet(data_dir / "dr16s82_ps1LCRaw.parquet")
   ztf = pd.read_parquet(data_dir / "dr16s82_ZuberLCRaw.parquet")

   sdss_objids = pd.Index(pd.read_parquet(data_dir / "dr16s82_sdssLCRaw.parquet", columns=["objectId"])["objectId"].unique())
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

   _, sdss_df = sdss_epochs()
   sdss_df = sdss_df.loc[sdss_df["objectId"].isin(cat_keys["objectId"])].assign(survey="sdss")

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
   )
   ztf_df = (ztf_df.assign(maglimit=ztf_maglimit(ztf_df))
             .loc[:, ["objectId", "band", "mjd", "mag", "magerr_p3", "maglimit"]]
             .rename(columns={"mjd": "time", "magerr_p3": "magerr"}))
   ztf_df["survey"] = "ztf"
   ztf_df["shallow"] = shallow_epochs(ztf_df["objectId"].to_numpy(), ztf_df["band"].to_numpy(), ztf_df["survey"].to_numpy(),
                                      ztf_df["mag"].to_numpy(), ztf_df["maglimit"].to_numpy())
   ztf_df = ztf_df.drop(columns="maglimit")

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
   df["shallow"] = df["shallow"].eq(True)
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


def _mad_sigma(x: np.ndarray) -> float:
   return 1.4826 * np.median(np.abs(x - np.median(x)))


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

SURVEY_SDSS_BAND = {"SERC-J/EJ": "g", "SERC-R/AAO-R": "r", "SERC-I": "i", "POSSI-E(S)": "r"}


def star_sky_positions() -> np.ndarray:
   with h5py.File(_ROOT / "data/roe_stars.hdf5") as f:
      return np.array([k.split("_") for k in f.keys()], dtype=float)


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


# lowest u = (limit - reference)/sigma validated on standards
# (10-20 pct to u = -2, star_null --plate-ccd); V held flat below
TRUNC_U_MIN = -2.0

CCD_TRUNC_CSV = _ROOT / "data/ccd_truncation_correction.csv"
CCD_BINS = np.arange(17.0, 23.51, 0.25)
CCD_MIN_OBJ = 60
CCD_BRIGHT = 19.5
CCD_BANDS = ["g", "r", "i", "z"]
CCD_MIN_FRAC = 0.2
# held-out test unbiased to here; beyond, g over-corrects (+0.072 mag at 22.6, +0.120 at 23.1)
CCD_VALID_MAX = 22.4
ZTF_META = _ROOT / "data/ztf/ztf_current_meta_sci.csv.gz"
ZTF_FAINT = 20.0
PS1_MAG_RANGE = (10.0, 25.0)
PALOMAR = EarthLocation.from_geodetic(-116.863 * u.deg, 33.356 * u.deg, 1706 * u.m)


def ztf_maglimit(z: pd.DataFrame) -> np.ndarray:
   # Zubercal mjd heliocentric mid-exposure; obsjd UTC start of the 30 s exposure; NaN unmatched
   m = pd.read_csv(ZTF_META)
   m = pd.DataFrame({"fieldid": m.field, "rcidin": (m.ccdid - 1) * 4 + m.qid - 1, "band": m.filtercode.str[1],
                     "t0": m.obsjd - 2400000.5, "maglimit": m.maglimit})
   if m.duplicated(["fieldid", "rcidin", "band", "t0"]).any():
       raise ValueError("duplicate ZTF exposure metadata")
   node = pd.DataFrame({"t": np.round(z["mjd"].to_numpy() * 8) / 8, "ra": np.round(z["objra"].to_numpy() * 2) / 2})
   nodes = node.drop_duplicates()
   nodes["ltt"] = Time(nodes.t.to_numpy(), format="mjd", scale="utc", location=PALOMAR).light_travel_time(
       SkyCoord(nodes.ra.to_numpy(), 0.0, unit="deg"), kind="heliocentric").to_value("day")
   q = pd.DataFrame({"fieldid": z["fieldid"].to_numpy(np.int64), "rcidin": z["rcidin"].to_numpy(np.int64),
                     "band": z["band"].to_numpy(str), "i": np.arange(len(z)),
                     "t": z["mjd"].to_numpy() - node.merge(nodes, how="left").ltt.to_numpy() - 15 / 86400})
   j = pd.merge_asof(q.sort_values("t"), m.sort_values("t0"), left_on="t", right_on="t0",
                     by=["fieldid", "rcidin", "band"], direction="nearest", tolerance=120 / 86400)
   return j.sort_values("i").maglimit.to_numpy()


def shallow_epochs(obj: np.ndarray, band: np.ndarray, survey: np.ndarray, mag: np.ndarray, maglimit: np.ndarray) -> np.ndarray:
   # limit brighter than median raw ZTF mag (ZTF note 8; unmatched kept), or median > ZTF_FAINT (Arevalo+2023)
   z = survey == "ztf"
   ref = pd.Series(np.where(z, mag, np.nan)).groupby([obj, band]).transform("median").to_numpy()
   matched = z & np.isfinite(maglimit)
   shallow = matched & (maglimit < ref)
   faint = z & (ref > ZTF_FAINT)
   drop = shallow | faint
   for b in sorted(set(band[z])):
       k = z & (band == b)
       print(f"{int(k.sum())} ztf {b} epochs, {int((matched & k).sum())} matched ({(matched & k).sum() / k.sum():.4f}), "
             f"{int((shallow & k).sum())} shallower than reference, "
             f"{int((faint & k).sum())} from {len(set(obj[faint & k]))} sources fainter than {ZTF_FAINT}, "
             f"{int((drop & k).sum())} dropped, {int((k & ~matched & ~drop).sum())} unmatched kept")
   return drop


def ccd_reference_mags() -> pd.Series:
   # qsogen catalog mags: deep, no epoch detection selection; raw SDSS median fills gaps
   cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet",
                         columns=["objectId"] + [f"sdss_{b}_qg" for b in CCD_BANDS])
   cat["OBJID"] = cat["objectId"].astype(str)
   qg = cat.set_index("OBJID")[[f"sdss_{b}_qg" for b in CCD_BANDS]]
   qg.columns = CCD_BANDS
   qg = qg.stack().rename("ref")
   qg.index = qg.index.set_names(["OBJID", "band"])
   raw = pd.read_parquet(_ROOT / "data/S82/dr16s82_sdssLCRaw.parquet", columns=["objectId", "filterID", "psMag"])
   raw = raw[raw["objectId"].isin(cat["objectId"]) & raw["filterID"].isin([filters[b] for b in bands_sdss])]
   sd = (raw["psMag"].astype(float).groupby([raw["objectId"].astype(str).rename("OBJID"),
                                             raw["filterID"].map({filters[b]: b for b in bands_sdss}).rename("band")])
         .median().rename("ref"))
   z0 = float(np.nanmedian((sd - qg).dropna()[qg.reindex(sd.index) < CCD_BRIGHT]))
   return (qg + z0).combine_first(sd)


def build_ccd_truncation(df_base: pd.DataFrame, ref: pd.Series,
                        out_csv: Path | None = CCD_TRUNC_CSV) -> pd.DataFrame:
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
                        ref: pd.Series) -> pd.DataFrame:
   df = df_base.copy()
   r = ref.reindex(pd.MultiIndex.from_arrays([df["OBJID"].astype(str), df["band"]])).to_numpy()
   off, v, fr = np.zeros(len(df)), np.ones(len(df)), np.ones(len(df))
   ok = np.isfinite(r)
   has = np.zeros(len(df), dtype=bool)
   sv, bd = df["survey"].to_numpy(), df["band"].to_numpy()
   for (s, b), g in curve.groupby(["survey", "band"], sort=True):
       m = (sv == s) & (bd == b)
       if not m.any():
           continue
       if not ok[m].all():
           raise ValueError(f"{s} {b} epoch without a CCD reference magnitude")
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


def redshift(d) -> np.ndarray:
   return np.where(d["Z_SYS"] > 0, d["Z_SYS"], d["Z_DR16Q"]).astype(float)


def redshifts(catalog_parquet=_ROOT / "data/S82/Catalog.parquet") -> pd.Series:
   cat = pd.read_parquet(catalog_parquet, columns=["objectId", "Z_SYS", "Z_DR16Q"])
   z = redshift(cat)
   if not (z > 0).all():
      raise ValueError("catalog quasar without a valid redshift")
   return pd.Series(z, index=cat.objectId.astype(str))


def match_groups_to_catalog(df_groups: pd.DataFrame, catalog_parquet: str, max_sep_arcsec: float = 1.0) -> pd.DataFrame:
   g = (
       df_groups[["GROUP_ID", "RA", "DEC"]]
       .dropna()
       .groupby("GROUP_ID", as_index=False)
       .median()
   )
   cat = pd.read_parquet(catalog_parquet, columns=["objectId", "RA", "DEC"])
   coords_cat = SkyCoord(ra=cat["RA"].to_numpy() * u.deg, dec=cat["DEC"].to_numpy() * u.deg, frame="icrs")
   coords_grp = SkyCoord(ra=g["RA"].to_numpy() * u.deg, dec=g["DEC"].to_numpy() * u.deg, frame="icrs")
   idx, sep2d, _ = coords_grp.match_to_catalog_sky(coords_cat)
   sep = sep2d.to(u.arcsec).value
   ok = sep <= float(max_sep_arcsec)
   oid = cat["objectId"].to_numpy()[idx[ok]].astype(str)
   return (
       pd.DataFrame({
           "GROUP_ID": g.loc[ok, "GROUP_ID"].to_numpy(),
           "OBJID": oid,
           "z": redshifts(catalog_parquet)[oid].to_numpy(),
           "sep_arcsec": sep[ok],
       })
       .sort_values("sep_arcsec")
       .drop_duplicates(subset=["GROUP_ID"], keep="first")
       .drop(columns=["sep_arcsec"])
       .reset_index(drop=True)
   )


def main() -> None:
   from plate_magnitude_models import native_longform
   df_base = concat_light_curves_df().rename(columns={"objectId": "OBJID"})
   if df_base["OBJID"].isna().any():
       raise ValueError("NaN OBJID rows after concat")
   df_base["OBJID"] = df_base["OBJID"].astype(str)
   ref = ccd_reference_mags()
   # CCD surveys detection limited (ZTF first): same treatment as plates before medians
   df_base = apply_ccd_truncation(df_base, build_ccd_truncation(df_base, ref), ref)
   bad = (df_base["survey"] == "ps1") & ~df_base["mag"].between(*PS1_MAG_RANGE)
   print(f"ps1 epochs outside {PS1_MAG_RANGE} mag dropped: {int(bad.sum())}")
   df_base = df_base[~bad]

   df_sss = native_longform()

   df_total = assemble_total_lightcurve(df_base, df_sss)

   n_possi = int((df_total["survey"] == "sss_possi").sum())
   native = pd.read_parquet(_ROOT / "data/plate_native_quasar_epochs.parquet", columns=["SURVEYNAME", "supported"])
   if n_possi != int((native.SURVEYNAME.eq("POSSI-E(S)") & native.supported).sum()):
       raise ValueError("Native POSS-I epoch count differs from its support ledger")
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
   out = _ROOT / "data/S82/total_lightcurves.parquet"
   df_total.to_parquet(out, index=False)
   print(f"rows {len(df_total)} objects {df_total['OBJID'].nunique()} sss {int(is_sss.sum())} "
         f"possi {n_possi} both {n_both} -> {out}")


if __name__ == "__main__":
   import argparse
   argparse.ArgumentParser().parse_args()
   main()
