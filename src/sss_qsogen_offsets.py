import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from qsogen import model_colours as mc
from assemble_lightcurves import ab_minus_vega, read_bandpass, redshift, sed_terms, vac_to_air
import ensemble_structure_function as esf

SSS_FILTERS = {
    "SSS_BJ": "serc-j.txt",
    "SSS_R": "serc-r.txt",
    "SSS_E": "possi-e.txt",
    "SSS_I": "serc-i.txt",
}
# photon-counting AB zeropoint const*simps(R/lam), as qsogen produce_zeropoints; 10 A resampling
for name, fname in SSS_FILTERS.items():
    w, T = read_bandpass(fname)
    fine = np.arange(w.min(), w.max() + 1e-9, 10.0)
    Tf = np.interp(fine, w, T)
    mc.wavarrs[name] = fine
    mc.resparrs[name] = Tf
    mc.zeropoints[name + "_AB"] = 0.1088544752 * mc.simps(Tf / fine, fine)

def _native_control_bands():
    from stellar_sed_calibration import bj_response

    # the calibration's own responses (only the quasar spectrum differs)
    bands = dict(bj=bj_response(), **{b: read_bandpass(f) for b, f in dict(r="iiiaf_og590.txt",
                 e="possi-e.txt", tp="techpan_og590.txt", g="SLOAN_SDSS.g.dat", r_sdss="SLOAN_SDSS.r.dat").items()})
    vega = {b: ab_minus_vega(*bands[b]) for b in ("bj", "r", "e", "tp")}
    return bands, vega


def observed_sed_control(manifest_path=None, output=None):
    """Native terms measured on archived SDSS spectra, beside the adopted qsogen terms"""
    from astropy.io import fits
    from stellar_sed_calibration import _weights

    root = _ROOT / "data/sdss_sed_control"
    manifest = pd.read_csv(manifest_path or root / "manifest.csv", dtype={"OBJID": str})
    # public manifest lacks OBJID, join the catalog on plate, MJD, fiber
    if "OBJID" not in manifest:
        keys = ["PLATE", "MJD", "FIBERID"]
        cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet", columns=[
            *keys, "objectId", "Z_SYS", "IF_BOSS_SDSS"]).set_index(keys, verify_integrity=True)
        cat = cat.assign(OBJID=cat.objectId.astype(str))
        manifest = manifest.join(cat.loc[pd.MultiIndex.from_frame(manifest[keys])].reset_index(drop=True))
    adopted = sed_terms(manifest.OBJID.unique(), ["bj", "r", "e", "tp", "sdss_g", "sdss_r"])
    bands, vega = _native_control_bands()
    rows = []
    for p in manifest.itertuples(index=False):
        with fits.open(root / p.file) as h:
            d, info = h[1].data, h[2].data
            # flux in erg/s/cm^2/A
            w, f, iv = 10. ** np.array(d["loglam"], float), np.array(d["flux"], float) * 1e-17, np.array(d["ivar"], float) / 1e-34
            warning = int(info["ZWARNING"][0])
            metadata = {key.lower(): info[key][0].item() if hasattr(info[key][0], 'item') else str(info[key][0])
                        for key in ['RUN2D', 'LAMBDA_EFF', 'BOSS_TARGET1', 'ANCILLARY_TARGET1',
                                    'ANCILLARY_TARGET2', 'EBOSS_TARGET0', 'EBOSS_TARGET1', 'EBOSS_TARGET2']
                        if key in info.names}
        w = vac_to_air(w)
        good = np.isfinite(f) & np.isfinite(iv) & (iv > 0)
        # interpolate over masked pixels (much masked weight or a long gap fails)
        flux = np.interp(w, w[good], f[good])
        variance = np.divide(1, iv, out=np.zeros(len(iv)), where=good)
        for native in ("bj", "r", "e", "tp"):
            curves = [bands["g" if native == "bj" else "r_sdss"], bands[native]]
            covered = all(w.min() < bp[0].min() and w.max() > bp[0].max() for bp in curves)
            row = dict(OBJID=p.OBJID, native=native, file=p.file, MJD=p.MJD, PLATE=p.PLATE,
                       FIBERID=p.FIBERID, instrument=p.IF_BOSS_SDSS, z=p.Z_SYS, color=adopted.at[p.OBJID, "sdss_g"] - adopted.at[p.OBJID, "sdss_r"],
                       zwarning=warning, full_coverage=covered, quality=False, **metadata)
            if covered:
                weights = np.array([_weights(w, *bp) for bp in curves])
                # same photon-weighted integral as abmag
                integrals = weights @ flux
                missing = np.max(weights[:, ~good].sum(axis=1) / weights.sum(axis=1))
                bad = np.flatnonzero((weights.sum(axis=0) > 0) & ~good)
                groups = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1)
                gap = max([w[g[-1]] - w[g[0]] for g in groups if len(g)] + [0.])
                delta = -2.5 * np.log10(integrals[0] / integrals[1]) + vega[native] - (0 if native == "bj" else .015)
                # formal ivar error for the 0.03 mag cut (no flux-calibration error)
                jac = 2.5 / np.log(10) * (weights[1] / integrals[1] - weights[0] / integrals[0])
                error = np.sqrt(np.sum(jac ** 2 * variance))
                row.update(observed=delta, adopted=adopted.at[p.OBJID, native], delta_observed=delta - adopted.at[p.OBJID, native],
                           formal_error=error, invalid_weight=missing, max_gap_angstrom=gap,
                           quality=bool(warning == 0 and error < .03 and missing < .005 and gap < 20))
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output or _ROOT / "data/quasar_observed_sed_control.csv", index=False)
    print(result[result.quality].groupby("native").delta_observed.quantile([.05, .5, .95]).to_string(), flush=True)
    return result


@lru_cache
def _light_curve_fluxes():
    from stellar_sed_calibration import ASINH_B

    raw = pd.read_parquet(_ROOT / "data/S82/dr16s82_sdssLCRaw.parquet", columns=["objectId", "psMag", "psMagErr", "filterID"]).query("filterID < 5")
    flux, error = psf_flux(raw.psMag, raw.psMagErr, ASINH_B[raw.filterID])
    key = [raw.objectId, raw.filterID]
    lc = pd.DataFrame(dict(flux=flux.groupby(key).median(), error=error.groupby(key).median(), n=flux.groupby(key).size()))
    lc["scatter"] = 1.4826 * abs(flux - flux.groupby(key).transform("median")).groupby(key).median()
    return lc[lc.n >= 5].unstack()


def psf_flux(mag, error, b):
    """Nanomaggies and their errors from SDSS asinh magnitudes (Lupton et al. 1999)"""
    arg, scale = -mag * np.log(10) / 2.5 - np.log(b), 2e9 * b
    return scale * np.sinh(arg), error * np.log(10) / 2.5 * scale * np.cosh(arg)


def light_curve_valid(objects, flux, error, k=5.):
    """SDSS bands whose PSF flux and error agree with the quasar's raw Stripe 82 light curve

    Deblended children carry zero fluxes or inflated errors that no SDSS flag marks
    Invalid beyond k robust scatters of the median flux or k times the median epoch error
    Returns objects by ugriz, True without five light-curve epochs
    """
    lc = _light_curve_fluxes().reindex(objects)
    median, scatter, epoch = (lc[c].to_numpy() for c in ("flux", "scatter", "error"))
    with np.errstate(invalid="ignore"):
        return ~((abs(flux - median) > k * np.hypot(scatter, epoch)) | (error > k * epoch))


def sdss_photometry(recovered=True):
    """SDSS PSF fluxes of every catalog quasar, one imaging epoch each

    DR16Q fluxes (Lyke et al. 2020) valid against the light curve, else the recovered DR16 detection
    Returns by OBJID
        z, RA, DEC, foreground_ebv, source
        flux_ and error_ per ugriz band in nanomaggies
    """
    from astropy.io import fits

    cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet",
                          columns=["objectId", "RA", "DEC", "PLATE", "MJD", "FIBERID", "Z_SYS", "Z_DR16Q"])
    key = lambda d: np.asarray(d["PLATE"], "i8") * 10 ** 9 + np.asarray(d["MJD"], "i8") * 10 ** 4 + np.asarray(d["FIBERID"], "i8")
    with fits.open(_ROOT / "data/dr16q_prop_May01_2024.fits", memmap=True) as h:
        keys = key(h[1].data)
        order = np.argsort(keys)
        rows = order[np.searchsorted(keys[order], key(cat)).clip(max=len(keys) - 1)]
        if not np.array_equal(keys[rows], key(cat)):
            raise ValueError("Catalog quasar missing from the DR16Q table")
        # EBV already 0.86 times SFD (Schlafly & Finkbeiner 2011)
        ebv, flux, ivar = (np.array(h[k].data[c][rows], float) for k, c in ((1, "EBV"), (2, "PSFFLUX"), (2, "PSFFLUX_IVAR")))
    names = [f"{k}_{b}" for k in ("flux", "error") for b in "ugriz"]
    p = pd.DataFrame(dict(z=redshift(cat), RA=cat.RA.to_numpy(), DEC=cat.DEC.to_numpy(), foreground_ebv=ebv, source="DR16Q"),
                     index=cat.objectId.rename("OBJID"))
    with np.errstate(invalid="ignore", divide="ignore"):
        error = ivar ** -.5
        bad = (flux == -9999) | ~(ivar > 0) | ~light_curve_valid(cat.objectId, flux, error)
        p[names] = np.where(np.tile(bad, 2), np.nan, np.hstack([flux, error]))
    if recovered:
        r = pd.read_csv(_ROOT / "data/quasar_sdss_recovered_photometry.csv", dtype={"OBJID": str, "photoObjID": str}).set_index("OBJID")
        p.loc[r.index, names] = r[names]
        p.loc[r.index, "source"] = r.source.str.cat(r.photoObjID, " ", na_rep="").str.strip()
    return p


class SdssSedModel:
    def __init__(self, z, foreground_ebv, grid_size=5001, extra=()):
        from stellar_sed_calibration import _weights, redden

        self.z = float(z)
        self.bands = {f"sdss_{b}": read_bandpass(f"SLOAN_SDSS.{b}.dat") for b in "ugriz"}
        self.bands.update({f"ps1_{b}": (mc.wavarrs[f"PS1_{b}"], mc.resparrs[f"PS1_{b}"]) for b in "griz"})
        native, self.vega = _native_control_bands()
        self.bands.update({b: native[b] for b in self.vega}, **dict(extra))
        # rest-frame grid past the host normalization and every response
        lower = min(890., min(w.min() for w, _ in self.bands.values()) / (1 + self.z) * .95)
        self.wave = np.geomspace(lower, 30000., grid_size)
        observed = self.wave * (1 + self.z)
        self.keep = (observed >= 2000) & (observed <= 33000)
        # Milky Way reddening once, on the model
        self.screen = redden(observed[self.keep], np.ones(self.keep.sum()), float(foreground_ebv))
        air = vac_to_air(observed[self.keep])
        self.weights = np.array([_weights(air, w, t) for w, t in self.bands.values()]) * 1e9

    # AB nanomaggies per response
    def __call__(self, parameters):
        from qsogen.qsosed import Quasar_sed

        sed = Quasar_sed(z=self.z, wavlen=self.wave, **dict(zip(("LogL3000", "ebv", "fragal"), parameters)))
        return self.weights @ (sed.flux[self.keep] * self.screen)

    def terms(self, flux):
        from stellar_sed_calibration import AB_OFFSET

        # zero model flux leaves the term undefined
        with np.errstate(invalid="ignore", divide="ignore"):
            m = dict(zip(self.bands, 22.5 - 2.5 * np.log10(flux)))
            # AB model magnitudes, AB SDSS minus PS1 as the CCD offsets
            out = {k: m[k] for k in self.bands if k[:4] == "sdss"}
            out.update({f"ps1_{b}": m[f"sdss_{b}"] - m[f"ps1_{b}"] for b in "griz"})
            # native Vega to catalog g or r (Thanjavur et al. 2021 offsets from AB)
            out.update({n: m[f"sdss_{b}"] - AB_OFFSET["ugriz".index(b)] - m[n] + self.vega[n] for n, b in zip(self.vega, "grrr")})
        return out


def _fit_sdss(row):
    from scipy.optimize import least_squares
    from stellar_sed_calibration import AB_OFFSET

    # LogL3000, intrinsic E(B-V) (negative is bluer than the default continuum), host fraction
    lo, hi = np.array([40., -.5, 0.]), np.array([48., 3., .99])
    scale = 10 ** (-.4 * AB_OFFSET)
    flux = np.array([row[f"flux_{b}"] for b in "ugriz"]) * scale
    # 0.03 mag floor for the fixed line strengths and continuum shape of the model
    error = np.hypot(np.array([row[f"error_{b}"] for b in "ugriz"]) * scale, .03 * np.log(10) / 2.5 * abs(flux))
    # u only below z = 2.3, before it drops out (qsogen README)
    use = np.isfinite(flux + error) & (np.arange(5) >= (row["z"] >= 2.3))
    out = dict(OBJID=row["OBJID"], z=row["z"], source=row["source"], bands="".join(np.array(list("ugriz"))[use]), dof=int(use.sum()) - 3)
    if out["dof"] < 1:
        return dict(out, status="fewer than four valid SDSS bands")
    model = SdssSedModel(row["z"], row["foreground_ebv"])
    p0 = np.array([45., .02, .1])
    f = model(p0)[:5][use]
    norm = np.sum(f * flux[use] / error[use] ** 2) / np.sum((f / error[use]) ** 2)
    if not norm > 0:
        return dict(out, status="no positive SDSS flux")
    p0[0] += np.log10(norm)
    best = least_squares(lambda p: (model(p)[:5][use] - flux[use]) / error[use], np.clip(p0, lo + 1e-6, hi - 1e-6),
                         bounds=(lo, hi), x_scale="jac")
    edge = np.minimum(best.x - lo, hi - best.x) < 1e-3
    flags = [name for name, hit in (("LogL3000 at bound", edge[0]), ("E(B-V) at bound", edge[1]),
                                    ("host fraction unconstrained", best.x[2] >= .98)) if hit]
    status = "; ".join(flags) or "converged" if best.success else best.message
    return dict(out, status=status, **dict(zip(("LogL3000", "intrinsic_ebv", "fragal"), best.x)),
                chi2=2 * best.cost, **model.terms(model(best.x)))


def fit_sdss_photometry():
    """qsogen fit (Temple et al. 2021) of each quasar's SDSS PSF fluxes, with the conversion terms of its best model"""
    rows = sdss_photometry().reset_index().to_dict("records")
    with Pool(8) as pool:
        result = pd.DataFrame(pool.imap(_fit_sdss, rows, chunksize=20))
    result.to_parquet(_ROOT / "data/quasar_sdss_sed_terms.parquet", index=False)
    print(result.status.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    esf.check_flags(__file__)
    if "--fit-sdss" in sys.argv:
        fit_sdss_photometry()
    elif "--observed-sed-control" in sys.argv:
        observed_sed_control()
