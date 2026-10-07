import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from qsogen import model_colours as mc
from assemble_lightcurves import ab_minus_vega, abmag, read_bandpass, vac_to_air
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

def _native_sed_row(args):
    from qsogen.qsosed import Quasar_sed
    from stellar_sed_calibration import redden

    p, bands, vega, wr, fr = args
    obj, z, logl, intrinsic_ebv, fragal, foreground = p
    model = Quasar_sed(z=z, LogL3000=logl, ebv=intrinsic_ebv, fragal=fragal)
    spectra = [(wr * (1 + z), fr / (1 + z)), (model.wavred, model.flux)]
    terms = []
    for w, f in spectra:
        use = (w >= 2000) & (w <= 33000)
        w, f = w[use], f[use]
        f = redden(w, f, foreground)
        w = vac_to_air(w)
        mags = {b: abmag(w, f, *bp) for b, bp in bands.items()}
        terms.append([mags["g" if b == "bj" else "r_sdss"] - mags[b] + vega[b]
                      - (0 if b == "bj" else .015) for b in ("bj", "r", "e", "tp")])
    return [dict(OBJID=obj, native=b, z=z, ebv=foreground, composite=terms[0][j],
                 qsogen=terms[1][j], delta_qsogen=terms[1][j] - terms[0][j])
            for j, b in enumerate(("bj", "r", "e", "tp"))]


def _native_control_bands():
    from plate_magnitude_models import NATIVE_ALPHA

    bands = {b: read_bandpass(f) for b, f in dict(bj="serc-j.txt", r="iiiaf_og590.txt",
             e="possi-e.txt", tp="techpan_og590.txt", g="SLOAN_SDSS.g.dat", r_sdss="SLOAN_SDSS.r.dat").items()}
    for b in ("bj", "e", "tp"):
        w, t = bands[b]
        bands[b] = w, t * (w / np.average(w, weights=t)) ** NATIVE_ALPHA[b]
    vega = {b: ab_minus_vega(*bands[b]) for b in ("bj", "r", "e", "tp")}
    return bands, vega


def native_sed_control():
    epochs = pd.read_parquet(_ROOT / "data/plate_native_quasar_epochs.parquet", columns=["OBJID", "z"])
    objects = epochs.drop_duplicates("OBJID").set_index("OBJID")
    objects.index = objects.index.astype(str)
    cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet").assign(
        OBJID=lambda d: d.objectId.astype(str)).set_index("OBJID")
    objects = objects.join(cat[["qg_LogL3000", "qg_ebv", "qg_fragal", "ebv"]]).dropna()
    objects["ebv"] *= .86
    bands, vega = _native_control_bands()
    composite = Table.read(_ROOT / "data/vandenberk_qso_composite_cds.txt", format="ascii.cds")
    wr, fr = np.asarray(composite["Wave"], float), np.asarray(composite["FluxD"], float)
    args = [(p, bands, vega, wr, fr) for p in objects.reset_index().itertuples(index=False, name=None)]
    with Pool(8) as pool:
        result = pool.map(_native_sed_row, args, chunksize=100)
    result = pd.DataFrame([row for group in result for row in group])
    result["valid"] = np.isfinite(result[["composite", "qsogen"]]).all(axis=1)
    result.to_csv(_ROOT / "data/quasar_native_sed_control.csv", index=False)
    print(result.groupby("native").delta_qsogen.quantile([.05, .5, .95]).to_string(), flush=True)


def observed_sed_control(manifest_path=None, output=None):
    from astropy.io import fits
    from stellar_sed_calibration import _weights

    root = _ROOT / "data/sdss_sed_control"
    manifest = pd.read_csv(manifest_path or root / "manifest.csv", dtype={"OBJID": str})
    if "OBJID" not in manifest:
        keys = ["PLATE", "MJD", "FIBERID"]
        cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet", columns=[
            *keys, "objectId", "Z_SYS", "IF_BOSS_SDSS", "sdss_g_qg", "sdss_r_qg"]).set_index(keys, verify_integrity=True)
        cat = cat.assign(OBJID=cat.objectId.astype(str), color=cat.sdss_g_qg - cat.sdss_r_qg)
        manifest = manifest.join(cat.loc[pd.MultiIndex.from_frame(manifest[keys])].reset_index(drop=True))
    control = pd.read_csv(_ROOT / "data/quasar_native_sed_control.csv", dtype={"OBJID": str}).set_index(["OBJID", "native"])
    bands, vega = _native_control_bands()
    rows = []
    for p in manifest.itertuples(index=False):
        with fits.open(root / p.file) as h:
            d, info = h[1].data, h[2].data
            w, f, iv = 10. ** np.array(d["loglam"], float), np.array(d["flux"], float) * 1e-17, np.array(d["ivar"], float) / 1e-34
            warning = int(info["ZWARNING"][0])
            metadata = {key.lower(): info[key][0].item() if hasattr(info[key][0], 'item') else str(info[key][0])
                        for key in ['RUN2D', 'LAMBDA_EFF', 'BOSS_TARGET1', 'ANCILLARY_TARGET1',
                                    'ANCILLARY_TARGET2', 'EBOSS_TARGET0', 'EBOSS_TARGET1', 'EBOSS_TARGET2']
                        if key in info.names}
        w = vac_to_air(w)
        good = np.isfinite(f) & np.isfinite(iv) & (iv > 0)
        flux = np.interp(w, w[good], f[good])
        variance = np.divide(1, iv, out=np.zeros(len(iv)), where=good)
        for native in ("bj", "r", "e", "tp"):
            curves = [bands["g" if native == "bj" else "r_sdss"], bands[native]]
            covered = all(w.min() < bp[0].min() and w.max() > bp[0].max() for bp in curves)
            row = dict(OBJID=p.OBJID, native=native, file=p.file, MJD=p.MJD, PLATE=p.PLATE,
                       FIBERID=p.FIBERID, instrument=p.IF_BOSS_SDSS, z=p.Z_SYS, color=p.color,
                       zwarning=warning, full_coverage=covered, quality=False, **metadata)
            if covered:
                weights = np.array([_weights(w, *bp) for bp in curves])
                integrals = weights @ flux
                missing = np.max(weights[:, ~good].sum(axis=1) / weights.sum(axis=1))
                bad = np.flatnonzero((weights.sum(axis=0) > 0) & ~good)
                groups = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1)
                gap = max([w[g[-1]] - w[g[0]] for g in groups if len(g)] + [0.])
                delta = -2.5 * np.log10(integrals[0] / integrals[1]) + vega[native] - (0 if native == "bj" else .015)
                jac = 2.5 / np.log(10) * (weights[1] / integrals[1] - weights[0] / integrals[0])
                error = np.sqrt(np.sum(jac ** 2 * variance))
                compared = control.loc[p.OBJID, native]
                row.update(observed=delta, composite=compared.composite, qsogen=compared.qsogen,
                           delta_observed=delta - compared.composite, delta_qsogen=compared.qsogen - compared.composite,
                           formal_error=error, invalid_weight=missing, max_gap_angstrom=gap,
                           quality=bool(warning == 0 and error < .03 and missing < .005 and gap < 20))
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output or _ROOT / "data/quasar_observed_sed_control.csv", index=False)
    print(result[result.quality].groupby("native").delta_observed.quantile([.05, .5, .95]).to_string(), flush=True)
    return result


if __name__ == "__main__":
    esf.check_flags(__file__)
    if "--observed-sed-control" in sys.argv:
        observed_sed_control()
    elif "--native-sed-control" in sys.argv:
        native_sed_control()
