import sys
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_structure_function as esf
from assemble_lightcurves import (plate_ccd_tie, build_plate_zeropoints, ivezic_star_mags,
                                  plate_zp_offset, star_zp_detections,
                                  truncation_curve,
                                  truncation_offset, truncation_variance)

_ROOT = Path(__file__).resolve().parents[1]
MIN_PAIRS = 200
CLIP_SIGMA = 5.0


def calibrated_detections(correct: bool = True, sat_cut: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    calib = pd.read_csv(_ROOT / "data/sss_plate_calibration.csv")
    df = star_zp_detections(calib, sat_cut)
    print(f"{df['star'].nunique()} stars, {len(df)} detections (all stars used, no subsampling)")
    zp_csv = _ROOT / "data/sss_plate_zeropoints.csv"
    zp = (pd.read_csv(zp_csv) if zp_csv.exists()
          else build_plate_zeropoints(df, str(zp_csv)))
    df = df.sort_values(["star", "bi", "mjd"], kind="stable").reset_index(drop=True)
    mp = df["SMAG"].to_numpy(dtype=float) - plate_zp_offset(
        zp, df["plate"], df["ra"], df["dec"])[0]
    if correct:
        ref = ivezic_reference(df["star"].to_numpy(), df["bi"].to_numpy())
        mp = (mp - truncation_offset(df["SURVEYNAME"], ref, truncation_curve())
              - plate_ccd_tie(df["plate"], ref))
    return df, mp


CCD_MJD = 53200.0  # mean epoch, 1998-2007 SDSS window
MAG_EDGES = np.arange(14.0, 22.51, 0.5)
MAG_CEN = 0.5 * (MAG_EDGES[:-1] + MAG_EDGES[1:])
CEDGES = np.arange(-0.5, 2.51, 0.1)
CCEN = 0.5 * (CEDGES[:-1] + CEDGES[1:])
SURVEY_SDSS_BAND = {"SERC-J/EJ": "g", "SERC-R/AAO-R": "r", "POSSI-E(S)": "r",
                    "SERC-I": "i"}


def _ivezic_star_mags() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, int]:
    m, v, n = ivezic_star_mags()
    return m["g"], m["r"], m["i"], v["g"], v["r"], v["i"], n


def ivezic_reference(star: np.ndarray, bi: np.ndarray) -> np.ndarray:
    gm, rm, im, _, _, _, npos = _ivezic_star_mags()
    if star.max() >= npos:
        raise ValueError("star index exceeds hdf5 key count")
    return np.choose(bi, [gm[star], rm[star], im[star]])


def _color_transform(d0: np.ndarray, col: np.ndarray, ccd: np.ndarray,
                     fit_sel: np.ndarray, linear: bool = False) -> np.ndarray:
    fs = fit_sel & np.isfinite(d0) & np.isfinite(col) & (ccd >= 14.0) & (ccd < 20.5)
    sl = np.polyfit(col[fs], d0[fs], 1)
    if linear:
        return d0 - np.polyval(sl, col)
    ci = np.digitize(col[fs], CEDGES) - 1
    meds = np.array([np.median(d0[fs][ci == k]) if (ci == k).sum() >= 500 else np.nan
                     for k in range(len(CCEN))])
    fin = np.isfinite(meds)
    print(f"  color relation: {meds[fin].min():+.3f} to {meds[fin].max():+.3f} over "
          f"g-r {CCEN[fin].min():.2f}-{CCEN[fin].max():.2f}; linear slope {sl[0]:+.3f}, "
          f"intercept {sl[1]:+.3f} ({fs.sum()} epochs)")
    return d0 - np.interp(col, CCEN[fin], meds[fin])


PAIR_SURVEYS = ["CCD", "SERC-J/EJ", "SERC-R/AAO-R", "POSSI-E(S)", "SERC-I"]
PAIR_BAND = {"SERC-J/EJ": "g", "SERC-R/AAO-R": "r", "POSSI-E(S)": "r", "SERC-I": "i"}
RESID_CSV = _ROOT / "data/plate_pair_residual_variance.csv"
NULL_MAG = np.array([17.5, 19.0, 20.5, 22.0])
NULL_LAG = np.logspace(1.0, 4.5, 11)


def _half(star: np.ndarray) -> np.ndarray:
    return (star.astype(np.int64) * 2654435761) % 2


def _fiducial_limits() -> pd.DataFrame:
    from plate_completeness import BRIGHT_BOUND
    s = pd.read_parquet(_ROOT / "data/plate_completeness_sample.parquet")
    return s[s.adopted_keep].groupby("band").reference_mag.agg(["min", "max"]).assign(min=BRIGHT_BOUND)


def star_residuals(remove_dropout: bool = False) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df, mp = calibrated_detections(correct=False)
    star, bi = df["star"].to_numpy(), df["bi"].to_numpy()
    gm, rm, im, ge2, re2, ie2, _ = _ivezic_star_mags()
    col = (gm - rm)[star]
    ccd = np.choose(bi, [gm[star], rm[star], im[star]])
    ce2 = np.choose(bi, [ge2[star], re2[star], ie2[star]])
    sv = df["SURVEYNAME"].astype(str).str.strip().to_numpy()
    res = np.full(len(df), np.nan)
    for name, fit in [("SERC-J/EJ", "SERC-J/EJ"), ("SERC-R/AAO-R", "SERC-R/AAO-R"),
                      ("POSSI-E(S)", "SERC-R/AAO-R"), ("SERC-I", "SERC-I")]:
        m = sv == name
        res[m] = _color_transform(mp - ccd, col, ccd, sv == fit)[m]
    curve = truncation_curve()
    offset = truncation_offset(sv, ccd, curve)
    if remove_dropout:
        for p in curve.itertuples():
            use = sv == p.survey
            x = np.clip(ccd[use], p.fit_lo, p.fit_hi) - 20
            offset[use] = p.a0 + p.a1*x + p.a2*x*x + p.a3*x*x*x
    return df, res - offset, ccd, ce2, sv


class NullPairs(NamedTuple):
    t: np.ndarray
    keep: np.ndarray
    half: np.ndarray
    star: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    w: np.ndarray
    A: np.ndarray
    dests: list[np.ndarray]
    nd: int
    nl: int
    npl: int


def _null_epochs(remove_dropout: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    df, res, ccd, ce2, sv = star_residuals(remove_dropout)
    v = truncation_variance(sv, df["plate"], df["ra"], df["dec"], ccd, truncation_curve(),
                            pd.read_csv(_ROOT / "data/sss_plate_zeropoints.csv"))
    e = pd.DataFrame(dict(star=df["star"].to_numpy(), band=pd.Series(sv).map(PAIR_BAND).to_numpy(),
                          code=pd.Series(sv).map(PAIR_SURVEYS.index).to_numpy(),
                          res=res - plate_ccd_tie(df["plate"], ccd), s2=df["SMAG_ERR"].to_numpy(float) ** 2,
                          v=np.ones(len(df)) if remove_dropout else v, mjd=df["mjd"].to_numpy(),
                          plate=df["plate"].to_numpy(np.int64), ccd=ccd, ce2=ce2, sv=sv,
                          s2cat=df["SMAG_ERR_CAT"].to_numpy(float) ** 2, raw=df["raw"].to_numpy(float),
                          ra=df["ra"].to_numpy(float), dec=df["dec"].to_numpy(float)))
    return e, v


def _null_pairs(e: pd.DataFrame, limits: pd.DataFrame | None, nsig: float = CLIP_SIGMA) -> NullPairs:
    e = e[np.isfinite(e.res) & np.isfinite(e.ccd)]
    c = e.drop_duplicates(["star", "band"]).assign(code=0, res=0.0, v=1.0, mjd=CCD_MJD, plate=-1)
    e = pd.concat([e, c.assign(s2=c.ce2)]).sort_values(["star", "band"], kind="stable")
    key = e.star.to_numpy() * 3 + e.band.map({"g": 0, "r": 1, "i": 2}).to_numpy()
    starts = np.flatnonzero(np.r_[True, np.diff(key) != 0])
    sizes = np.diff(np.r_[starts, len(e)])
    ii, jj = [], []
    for a, b in combinations(range(int(sizes.max())), 2):
        s = starts[sizes > b]
        ii.append(s + a)
        jj.append(s + b)
    i, j = np.concatenate(ii), np.concatenate(jj)
    col = lambda k: e[k].to_numpy()
    code, pl = col("code"), col("plate")
    ca, cb = np.minimum(code[i], code[j]), np.maximum(code[i], code[j])
    dm, se2 = col("res")[i] - col("res")[j], col("s2")[i] + col("s2")[j]
    w = 2.0 / (col("v")[i] + col("v")[j])
    mag = col("ccd")[i]
    nm = len(MAG_CEN)
    km = np.searchsorted(MAG_EDGES, mag, side="right") - 1
    t = np.where((km >= 0) & (km < nm), (ca * len(PAIR_SURVEYS) + cb) * nm + km, -1)
    g = pd.Series(dm).groupby(t)
    dev = np.abs(dm - g.transform("median").to_numpy())
    keep = (t >= 0) & (dev <= nsig * 1.4826 * pd.Series(dev).groupby(t).transform("median"))
    plates = np.unique(pl[pl >= 0])
    pidx = lambda q: np.where(q >= 0, np.searchsorted(plates, q), -1)
    bnd = pd.Series(col("band")[i]).map({"g": 0, "r": 1}).fillna(-1).to_numpy(int)
    kd = np.searchsorted(NULL_MAG, mag, side="right") - 1
    kl = np.searchsorted(NULL_LAG, np.abs(col("mjd")[i] - col("mjd")[j]), side="right") - 1
    nd = 2 * (len(NULL_MAG) - 1) + (0 if limits is None else 2)
    nl = len(NULL_LAG) - 1
    d = np.where((bnd >= 0) & (kd >= 0) & (kd < len(NULL_MAG) - 1) & (kl >= 0) & (kl < nl),
                 (bnd * (len(NULL_MAG) - 1) + kd) * nl + kl, -1)
    destinations = [d]
    if limits is not None:
        adopted = np.full(len(d), -1)
        for bi_, b in enumerate("gr"):
            keep_adopted = (bnd == bi_) & (mag >= limits.at[b, "min"]) & (mag <= limits.at[b, "max"]) & (kl >= 0) & (kl < nl)
            adopted[keep_adopted] = (nd - 2 + bi_) * nl + kl[keep_adopted]
        destinations.append(adopted)
    return NullPairs(t, np.asarray(keep), _half(col("star")[i]), col("star")[i], pidx(pl[i]), pidx(pl[j]), w,
                     w * (dm ** 2 - se2), destinations, nd, nl, len(plates))


def _null_tables(P: NullPairs, keep: np.ndarray, min_pairs: int = MIN_PAIRS, u: np.ndarray | float = 1.0,
                 star_weights: bool = False, jack: bool = True) -> tuple[pd.DataFrame, dict]:
    t, half, p1, p2, nd, nl, npl = P.t, P.half, P.p1, P.p2, P.nd, P.nl, P.npl
    nm = len(MAG_CEN)
    nt = len(PAIR_SURVEYS) ** 2 * nm
    ok_t = np.stack([np.bincount(t[keep & (half == h)], minlength=nt) for h in (0, 1)]).min(axis=0) >= min_pairs / 2
    keep = keep & ok_t[np.maximum(t, 0)]
    w, A, un = P.w * u, P.A * u, np.broadcast_to(u, t.shape)
    wd, Ad = w, A
    if star_weights:
        m = keep & (P.dests[-1] >= 0)
        n = pd.Series(1, index=np.flatnonzero(m)).groupby([P.star[m], P.dests[-1][m]]).transform("size")
        ud = np.zeros(len(t))
        ud[n.index] = 1.0 / n.to_numpy()
        wd, Ad = w * ud, A * ud
    TW, TA, TN = (np.zeros((2, npl + 1, nt)) for _ in range(3))
    DW, DA = np.zeros((2, npl + 1, nd * nl, nt)), np.zeros((2, npl + 1, nd * nl))
    for q, extra in [(np.full(len(t), npl), None), (p1, p1 >= 0), (p2, (p2 >= 0) & (p2 != p1))][:3 if jack else 1]:
        m = keep & (np.ones(len(t), bool) if extra is None else extra)
        for h in (0, 1):
            k = m & (half == h)
            np.add.at(TW[h], (q[k], t[k]), w[k])
            np.add.at(TA[h], (q[k], t[k]), A[k])
            np.add.at(TN[h], (q[k], t[k]), un[k])
            for dest in P.dests:
                valid = k & (dest >= 0)
                np.add.at(DW[h], (q[valid], dest[valid], t[valid]), wd[valid])
                np.add.at(DA[h], (q[valid], dest[valid]), Ad[valid])
    full = lambda X: X[:, npl:npl + 1]
    loo = lambda X: full(X) - X[:, :npl]

    def null(TW, TA, DW, DA):
        with np.errstate(invalid="ignore", divide="ignore"):
            F = np.where(TW > 0, TA / TW, 0.0)
            den = DW.sum(axis=(0, 3))
            return DA.sum(axis=0) / den, (DA.sum(axis=0) - np.einsum("hpdt,hpt->pd", DW, F[::-1])) / den

    pooled = lambda X: X.reshape(*X.shape[:2], nd, nl, *X.shape[3:]).sum(axis=3)
    out = {}
    for tag, dw, da in [("lag", DW, DA), ("all", pooled(DW), pooled(DA))]:
        b0, a0 = null(full(TW), full(TA), full(dw), full(da))
        bj, aj = null(loo(TW), loo(TA), loo(dw), loo(da))
        used = (dw[:, :npl].sum(axis=(0, 3)) > 0)
        n = used.sum(axis=0)
        err = []
        for th in (bj, aj):
            th = np.where(used, th, np.nan)
            with np.errstate(invalid="ignore"):
                jk = np.sqrt((n - 1) / np.maximum(n, 1) * np.nansum((th - np.nanmean(th, axis=0)) ** 2, axis=0))
            err.append(jk)
        out[tag] = (b0[0], a0[0], err[0], err[1], full(dw)[:, 0].sum(axis=(0, 2)), n, aj)

    rows = []
    with np.errstate(invalid="ignore", divide="ignore"):
        FT, NT = full(TA)[:, 0].sum(axis=0) / full(TW)[:, 0].sum(axis=0), TN[:, npl].sum(axis=0)
        FJ = loo(TA).sum(axis=0) / loo(TW).sum(axis=0)
    usedt = TN[:, :npl].sum(axis=0) > 0
    for k in np.flatnonzero((NT >= min_pairs) & jack):
        cl, mk = divmod(k, nm)
        sa, sb = PAIR_SURVEYS[cl // len(PAIR_SURVEYS)], PAIR_SURVEYS[cl % len(PAIR_SURVEYS)]
        th = FJ[usedt[:, k], k]
        rows.append(dict(survey_a=sa, survey_b=sb, band=PAIR_BAND[sb], mag_lo=MAG_EDGES[mk],
                         mag_hi=MAG_EDGES[mk + 1], sf2_mag2=FT[k],
                         sf2_jack_err=float(np.sqrt((len(th) - 1) / len(th) * np.sum((th - th.mean()) ** 2))),
                         n_pairs=int(NT[k]), n_plates=len(th)))
    return pd.DataFrame(rows), out


def _adopted_bins(out: dict, nd: int, nl: int, bi: int) -> np.ndarray:
    _, a0, _, _, nw, npl_c, _ = out["lag"]
    dd = (nd - 2 + bi) * nl + np.arange(nl)
    return dd[(nw[dd] > 0) & np.isfinite(a0[dd]) & (npl_c[dd] >= 3)]


def main(remove_dropout: bool = True) -> None:
    if remove_dropout and (_ROOT / "data/plate_native_calibration.json").exists():
        import plate_magnitude_models as native
        native.NATIVE_VARIANT="effective"
        native.NATIVE_WORK=native.NATIVE_BASE/"effective"
        native.native_null()
        return
    e, v = _null_epochs(remove_dropout)
    if remove_dropout:
        sv, ccd = e.sv.to_numpy(), e.ccd.to_numpy()
        rows, selected_limits = [], _fiducial_limits()
        for p in truncation_curve().itertuples():
            if p.band not in ("g", "r"):
                continue
            lo, hi = selected_limits.loc[p.band]
            use = (sv == p.survey) & (ccd >= lo) & (ccd <= hi)
            m = np.clip(ccd[use], p.fit_lo, p.fit_hi)
            u = (p.mag_limit - m) / p.sigma
            gaussian = -p.sigma * np.exp(norm.logpdf(u) - norm.logcdf(u))
            rows.append(dict(survey=p.survey, band=p.band, mag_lo=lo, mag_hi=hi,
                             n_detections=int(use.sum()), max_gaussian_mag=float(np.max(np.abs(gaussian))),
                             min_old_variance=float(v[use].min()), mean_old_variance=float(v[use].mean())))
        pd.DataFrame(rows).to_csv(esf.cache_path("star_calibration_compatibility.csv"), index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    P = _null_pairs(e, selected_limits if remove_dropout else None)
    tab, out = _null_tables(P, P.keep)
    nd, nl = P.nd, P.nl
    output = _ROOT / ("data" if remove_dropout else "temp/pooled_diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    tab.to_csv(output / RESID_CSV.name, index=False)
    for (sa, sb), g in tab.groupby(["survey_a", "survey_b"], sort=False):
        print(f"residual {sb} x {sa}: " + " ".join(
            f"{r.mag_lo:.1f} {1e3 * r.sf2_mag2:+.2f}+-{1e3 * r.sf2_jack_err:.2f}" for r in g.itertuples())
              + " (1e-3 mag^2)")

    b0, a0, eb, ea, nw, npl_c, _ = out["lag"]
    B0, A0, EB, EA, _, _, _ = out["all"]
    lagc = np.sqrt(NULL_LAG[:-1] * NULL_LAG[1:])
    if remove_dropout:
        adopted_rows = []
        for bi_, b in enumerate("gr"):
            dd = _adopted_bins(out, nd, nl, bi_)
            x2 = np.sum((a0[dd] / ea[dd]) ** 2)
            lo, hi = selected_limits.loc[b]
            print(f"adopted-range null {b} {lo:.6f}-{hi:.6f}: diagonal chi2 {x2:.3f} / "
                  f"{len(dd)} bins, approximate p {chi2.sf(x2, len(dd)):.4f}")
            adopted_rows.extend(dict(band=b, mag_lo=lo, mag_hi=hi,
                                    observed_days=lagc[k % nl], sf2_before=b0[k], sf2_after=a0[k],
                                    error_before=eb[k], error_after=ea[k], n_plates=int(npl_c[k])) for k in dd)
        pd.DataFrame(adopted_rows).to_csv(output / "standard_star_adopted_null_structure_function.csv", index=False)
    null_rows = []
    for bi_, b in enumerate("gr"):
        for kd_ in range(len(NULL_MAG) - 1):
            dd = (bi_ * (len(NULL_MAG) - 1) + kd_) * nl + np.arange(nl)
            f = (nw[dd] > 0) & np.isfinite(a0[dd]) & (npl_c[dd] >= 3)
            if np.any(~np.isfinite(ea[dd][f]) | (ea[dd][f] <= 0)):
                raise ValueError("Undefined plate jackknife uncertainty")
            null_rows.extend(dict(band=b, mag_lo=NULL_MAG[kd_], mag_hi=NULL_MAG[kd_ + 1],
                observed_days=lagc[q], sf2_before=b0[dd][q], sf2_after=a0[dd][q],
                error_before=eb[dd][q], error_after=ea[dd][q], n_plates=int(npl_c[dd][q]))
                for q in np.flatnonzero(f))
            pk = bi_ * (len(NULL_MAG) - 1) + kd_
            x2 = float(np.sum((a0[dd][f] / ea[dd][f]) ** 2))
            print(f"null {b} {NULL_MAG[kd_]:.1f}-{NULL_MAG[kd_ + 1]:.1f}: before {1e3 * B0[pk]:+.2f}+-{1e3 * EB[pk]:.2f}, "
                  f"after {1e3 * A0[pk]:+.2f}+-{1e3 * EA[pk]:.2f} (1e-3 mag^2, all lags, plate jackknife); "
                  f"diagonal chi2 vs zero over {f.sum()} lag bins {x2:.1f}, approximate p {chi2.sf(x2, f.sum()):.3f}; per lag "
                  + " ".join(f"{lagc[q]:.0f}d {1e3 * b0[dd][q]:+.1f}/{1e3 * a0[dd][q]:+.1f}+-{1e3 * ea[dd][q]:.1f}"
                             f"({npl_c[dd][q]})" for q in np.flatnonzero(f)))
    pd.DataFrame(null_rows).to_csv(output / "standard_star_null_structure_function.csv", index=False)


if __name__ == "__main__":
    esf.check_flags(__file__)
    main()
