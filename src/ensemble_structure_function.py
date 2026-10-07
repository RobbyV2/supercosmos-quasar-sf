from itertools import combinations
from pathlib import Path
from typing import Literal
import ast
import re
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "temp/cache"
BANDS = ["g", "r"]
EDGES = np.logspace(0.75, 4.5, 16)
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
NB = len(EDGES) - 1
# Hartlap factor 0.998 here (0.920 at 200); own generator keeps 200-draw error bars fixed
N_COV_BOOT = 10000
CLIP_SIGMA = 5.0
SEGMENT = ["OBJID", "band", "survey"]
QVC_REJECT = False
QVC_HALF_WINDOW = 30.0
QVC_MIN_NEIGHBORS = 8
QVC_SIGMA = 4.0

k = int(np.searchsorted(EDGES, 7300.0 / (1.0 + 1.0), side="right")) - 1
if not (np.isclose(EDGES[k], 10**3.5) and np.isclose(EDGES[k + 1], 10**3.75)):
    raise AssertionError("rest-frame lag binning check failed")

LC_PARQUET = "data/S82/total_lightcurves.parquet"


def cache_path(name: str) -> Path:
    path = CACHE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cli_flags(path) -> set[str]:
    return {n.value for n in ast.walk(ast.parse(Path(path).read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and re.fullmatch(r"--[a-z][\w-]*", n.value)}


def check_flags(path) -> None:
    if bad := sorted({a.split("=", 1)[0] for a in sys.argv[1:] if a.startswith("--")} - cli_flags(path)):
        raise SystemExit(f"unknown option {' '.join(bad)}")


def condense_nightly(df: pd.DataFrame) -> pd.DataFrame:
    if not np.isfinite(df["magerr"]).all() or (df["magerr"] <= 0).any():
        raise ValueError("non-positive or non-finite magerr in light curves")
    df = df.assign(
        night=np.floor(df["time"].to_numpy()).astype(np.int64),
        w=df["magerr"].to_numpy() ** -2.0,
        is_sss=df["survey"].str.startswith("sss"),
        plate=df["plate"] if "plate" in df.columns else -1,
        ok=df["calib_ok"] if "calib_ok" in df.columns else True,
        vt=df["trunc_v"] if "trunc_v" in df.columns else 1.0,
    )
    df["wm"] = df["w"] * df["mag"]
    g = df.groupby(["OBJID", "band", "night"], sort=False, observed=True).agg(
        wsum=("w", "sum"), wmsum=("wm", "sum"), sss=("is_sss", "max"), survey=("survey", "first"),
        plate=("plate", "max"), calib_ok=("ok", "min"), trunc_v=("vt", "min"))
    g["mag"] = g["wmsum"] / g["wsum"]
    g["sig"] = g["wsum"] ** -0.5
    return g.reset_index()[["OBJID", "band", "night", "mag", "sig", "sss", "survey", "plate",
                            "calib_ok", "trunc_v"]]


def segment_deviations(nightly: pd.DataFrame) -> pd.DataFrame:
    g = nightly.groupby(SEGMENT, observed=True, sort=False)
    res = nightly["mag"] - g["mag"].transform("median")
    mad = 1.4826 * res.abs().groupby([nightly[c] for c in SEGMENT], observed=True, sort=False).transform("median")
    return nightly.assign(res=res, mad=mad, err=g["sig"].transform("median"))


def reject_outliers(nightly: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    d, sss = segment_deviations(nightly), nightly["sss"].to_numpy(bool)
    keep = (np.abs(d["res"].to_numpy()) <= CLIP_SIGMA * np.maximum(d["mad"].to_numpy(), d["err"].to_numpy())) | sss
    return nightly.loc[keep], int((~keep).sum()), len(nightly), int((~keep & sss).sum())


def qvc_significance(nightly: pd.DataFrame) -> np.ndarray:
    df = nightly.sort_values(["OBJID", "band", "night"], kind="stable")
    m, e = df["mag"].to_numpy(), df["sig"].to_numpy()
    g = df.groupby(["OBJID", "band"], sort=False, observed=True).ngroup().to_numpy()
    key = df["night"].to_numpy(dtype=float) + g * 1e7
    lo = np.searchsorted(key, key - QVC_HALF_WINDOW, side="left")
    hi = np.searchsorted(key, key + QVC_HALF_WINDOW, side="right")
    rows = np.flatnonzero(hi - lo - 1 >= QVC_MIN_NEIGHBORS)
    off = np.arange(int((hi - lo).max()))
    sig = np.full(len(df), np.nan)
    for a in range(0, len(rows), 200_000):
        r = rows[a:a + 200_000]
        idx = lo[r, None] + off
        ok = (idx < hi[r, None]) & (idx != r[:, None])
        nb = np.where(ok, m[np.minimum(idx, len(df) - 1)], np.nan)
        med = np.nanmedian(nb, axis=1)
        mad = np.nanmedian(np.abs(nb - med[:, None]), axis=1)
        sig[r] = np.abs(m[r] - med) / np.hypot(e[r], 1.4826 * mad)
    out = np.full(len(nightly), np.nan)
    out[nightly.index.get_indexer(df.index)] = sig
    return out


def qvc_mask(nightly: pd.DataFrame, threshold: float = QVC_SIGMA) -> np.ndarray:
    return qvc_significance(nightly) > threshold


def clean_nightly(lc: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    ids = lc["OBJID"].unique()
    parts = []
    for chunk in np.array_split(ids, max(1, len(ids) // 5000)):
        n = condense_nightly(lc[lc["OBJID"].isin(chunk)])
        if QVC_REJECT:
            n = n.loc[~qvc_mask(n)]
        out, *rest = reject_outliers(n)
        parts.append((pd.concat([out, floor_variance(out)], axis=1), *rest))
    return (pd.concat([q[0] for q in parts], ignore_index=True),
            *(sum(q[k] for q in parts) for k in (1, 2, 3)))


FLOOR_SURVEY = {("g", False): "SERC-J/EJ", ("r", False): "SERC-R/AAO-R",
                ("r", True): "POSSI-E(S)", ("i", False): "SERC-I"}
PAIR_SURVEYS = ["CCD", "SERC-J/EJ", "SERC-R/AAO-R", "POSSI-E(S)", "SERC-I", "TechPan"]
FV_COLS = [f"fv{k}" for k in range(len(PAIR_SURVEYS))]
HIST_EDGES = np.linspace(-4.0, 4.0, 4001)


def floor_variance(nightly: pd.DataFrame, col: str = "sf2_mag2") -> pd.DataFrame:
    # residual plate variance per epoch and partner survey, at CCD reference mag (nearest bin);
    # col="sf2_jack_err" gives its error
    tab = pd.read_csv(_ROOT / "data/plate_native_pair_variance.csv")
    names = dict(ccd="CCD", bj="SERC-J/EJ", r="SERC-R/AAO-R", e="POSSI-E(S)", tp="TechPan")
    tab = tab.assign(survey_a=tab.native_a.map(names), survey_b=tab.native_b.map(names))
    tab = tab.rename(columns={"residual_variance": "sf2_mag2", "training_floor_se": "sf2_jack_err"})
    if nightly.band.eq("i").any():
        old = pd.read_csv(_ROOT / "data/plate_pair_residual_variance.csv")
        tab = pd.concat([tab, old[old.survey_a.eq("SERC-I") | old.survey_b.eq("SERC-I")]], ignore_index=True)
    sss = nightly["sss"].to_numpy(dtype=bool)
    ref = (nightly[~sss].groupby(["OBJID", "band"], observed=True)["mag"].median()
           .reindex(pd.MultiIndex.from_arrays([nightly["OBJID"], nightly["band"]])).to_numpy(float))
    possi = nightly["survey"].eq("sss_possi").to_numpy()
    code = np.zeros(len(nightly), dtype=np.int8)
    for (b, p), name in FLOOR_SURVEY.items():
        code[sss & nightly["band"].eq(b).to_numpy() & (possi == p)] = PAIR_SURVEYS.index(name)
    code[sss & nightly.band.eq("r").to_numpy() & nightly.plate.isin([131900, 131903]).to_numpy()] = PAIR_SURVEYS.index("TechPan")
    out = np.zeros((len(nightly), len(PAIR_SURVEYS)), dtype=np.float32)
    for (sa, sb), t in tab.groupby(["survey_a", "survey_b"]):
        t = t.sort_values("mag_lo")
        cen = 0.5 * (t["mag_lo"] + t["mag_hi"]).to_numpy()
        v = t[col].to_numpy()
        f = np.where(np.isfinite(ref), v[np.searchsorted(0.5 * (cen[1:] + cen[:-1]), ref)],
                     np.median(v))
        ia, ib = PAIR_SURVEYS.index(sa), PAIR_SURVEYS.index(sb)
        for x, y in ((ia, ib), (ib, ia)):
            out[code == x, y] = f[code == x]
    return pd.DataFrame(out, columns=FV_COLS, index=nightly.index).assign(pcode=code)


def group_pairs(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    sizes = np.diff(np.r_[starts, len(key)])
    i, j = zip(*[(s + a, s + b) for a, b in combinations(range(int(sizes.max())), 2) for s in [starts[sizes > b]]])
    return np.concatenate(i), np.concatenate(j)


def accumulate(nightly: pd.DataFrame, objids: np.ndarray, zmap: pd.Series,
               edges: np.ndarray = EDGES, plate_acc: np.ndarray | None = None,
               plate_pos: dict[int, int] | None = None,
               bands: list[str] | None = None, resid: bool = False,
               dm_hist: np.ndarray | None = None) -> np.ndarray:
    bands = BANDS if bands is None else bands
    nb = len(edges) - 1
    if "pcode" not in nightly.columns:
        nightly = pd.concat([nightly, floor_variance(nightly)], axis=1)
    acc = np.zeros((len(objids), len(bands), nb, 6))
    oi_map = {o: i for i, o in enumerate(objids)}
    bi_map = {b: i for i, b in enumerate(bands)}
    for (o, b), sub in nightly.groupby(["OBJID", "band"], sort=False, observed=True):
        n = len(sub)
        if n < 2 or b not in bi_map:
            continue
        z = zmap[o]
        t = sub["night"].to_numpy(dtype=float)
        m = sub["mag"].to_numpy()
        s2 = sub["sig"].to_numpy() ** 2
        fl = sub["sss"].to_numpy(dtype=bool)
        vt = (sub["trunc_v"].to_numpy(dtype=float) if "trunc_v" in sub.columns
              else np.ones(n))
        i, j = np.triu_indices(n, k=1)
        dt = np.abs(t[i] - t[j]) / (1.0 + z)
        idx = np.searchsorted(edges, dt, side="right") - 1
        sel = (idx >= 0) & (idx < nb)
        idx = idx[sel]
        # retained-variance factor, plate and CCD epochs
        w = 2.0 / (vt[i[sel]] + vt[j[sel]])
        dm2 = w * (m[i[sel]] - m[j[sel]]) ** 2
        # residual plate variance: plate-CCD pair takes its survey term, plate-plate the survey-pair term
        fvar = w * sub[FV_COLS].to_numpy(dtype=float)[i[sel], sub["pcode"].to_numpy()[j[sel]]]
        se2 = w * (s2[i[sel]] + s2[j[sel]]) + (fvar if resid else 0.0)
        anysss = w * (fl[i[sel]] | fl[j[sel]])
        bi = bi_map[b]
        a = acc[oi_map[o], bi]
        a[:, 0] += np.bincount(idx, minlength=nb)
        a[:, 1] += np.bincount(idx, weights=dm2, minlength=nb)
        a[:, 2] += np.bincount(idx, weights=se2, minlength=nb)
        a[:, 3] += np.bincount(idx, weights=anysss, minlength=nb)
        a[:, 4] += np.bincount(idx, weights=fvar, minlength=nb)
        # slot 5: summed rest-frame lag, for each bin's pair-weighted mean lag
        a[:, 5] += np.bincount(idx, weights=dt[sel], minlength=nb)
        if dm_hist is not None:
            hc = np.clip(np.searchsorted(HIST_EDGES, m[i[sel]] - m[j[sel]]) - 1, 0, len(HIST_EDGES) - 2)
            np.add.at(dm_hist[bi], (idx, hc), np.stack([np.ones_like(dm2), dm2, se2], axis=1))
        if plate_acc is not None and fl.any():
            code = np.array([plate_pos.get(int(p), -1) for p in sub["plate"].to_numpy()])
            pi, pj = code[i[sel]], code[j[sel]]
            for c, msk in [(pi, pi >= 0), (pj, (pj >= 0) & (pj != pi))]:
                np.add.at(plate_acc[:, bi, :, 0], (c[msk], idx[msk]), 1.0)
                np.add.at(plate_acc[:, bi, :, 1], (c[msk], idx[msk]), dm2[msk])
                np.add.at(plate_acc[:, bi, :, 2], (c[msk], idx[msk]), se2[msk])
    return acc


KERNELS = (accumulate, clean_nightly, condense_nightly, segment_deviations, reject_outliers, floor_variance)


SFMethod = Literal["object_mean", "object_median", "positive_median", "amplitude_mean", "pooled"]


def object_second_moments(acc: np.ndarray) -> np.ndarray:
    n = acc[..., 0]
    return np.divide(acc[..., 1] - acc[..., 2], n,
                     out=np.full_like(n, np.nan), where=n > 0)


def ensemble_second_moment(acc: np.ndarray, method: SFMethod = "object_mean",
                           weights: np.ndarray | None = None) -> np.ndarray:
    v = object_second_moments(acc)
    valid = np.isfinite(v)
    w = np.ones_like(v) if weights is None else np.broadcast_to(weights, v.shape)
    if np.any(~np.isfinite(w) | (w < 0)):
        raise ValueError("object weights must be finite and nonnegative")
    if method == "pooled":
        w = w * acc[..., 0]
    elif method in ("object_median", "positive_median"):
        if weights is not None:
            raise ValueError("median aggregation requires equal object weights")
        if method == "positive_median":
            return np.nanmedian(np.sqrt(np.where(v > 0, v, np.nan)), axis=0) ** 2
        return np.nanmedian(v, axis=0)
    elif method == "amplitude_mean":
        valid &= v > 0
        v = np.sqrt(np.maximum(v, 0))
    elif method != "object_mean":
        raise ValueError(f"unknown SF method {method}")
    num = np.where(valid, v * w, 0.0).sum(axis=0)
    den = np.where(valid, w, 0.0).sum(axis=0)
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out ** 2 if method == "amplitude_mean" else out


def object_bootstrap(acc: np.ndarray, n_boot: int = N_COV_BOOT, seed: int = 42,
                     weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    v = object_second_moments(acc)
    w = np.ones_like(v) if weights is None else np.broadcast_to(weights, v.shape)
    if np.any(~np.isfinite(w) | (w < 0)):
        raise ValueError("object weights must be finite and nonnegative")
    den = np.where(np.isfinite(v), w, 0.0).reshape(len(acc), -1)
    num = np.nan_to_num(v * w, nan=0.0).reshape(len(acc), -1)
    out = np.full((n_boot, num.shape[1]), np.nan)
    for r in range(0, n_boot, 128):
        size = min(128, n_boot - r)
        ids = rng.integers(0, len(acc), (size, len(acc)))
        counts = np.bincount((ids + np.arange(size)[:, None] * len(acc)).ravel(),
                             minlength=size * len(acc)).reshape(size, len(acc))
        ns, ds = counts @ num, counts @ den
        out[r:r + size] = np.divide(ns, ds, out=np.full_like(ns, np.nan), where=ds > 0)
    out = out.reshape((n_boot,) + acc.shape[1:-1])
    bounds = np.nanpercentile(out, [16, 50, 84], axis=0)
    return bounds, out


def ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    mu = np.trace(S) / p
    d2 = ((S - mu * np.eye(p)) ** 2).sum()
    b2 = min(((np.einsum("ni,nj->nij", Xc, Xc) - S) ** 2).sum() / n ** 2, d2)
    rho = 0.0 if d2 == 0.0 else b2 / d2
    return (1.0 - rho) * S + rho * mu * np.eye(p), rho, S


def load_properties(objids: np.ndarray) -> pd.DataFrame:
    cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet",
                          columns=["objectId", "PLATE", "MJD", "FIBERID"])
    cat["OBJID"] = cat["objectId"].astype(str)
    with fits.open(_ROOT / "data/dr16q_prop_May01_2024.fits", memmap=True) as hdul:
        d = hdul[1].data
        prop = pd.DataFrame({
            "PLATE": d["PLATE"].astype(np.int64),
            "MJD": d["MJD"].astype(np.int64),
            "FIBERID": d["FIBERID"].astype(np.int64),
            "LOGMBH": d["LOGMBH"].astype(np.float64),
            "LOGLEDD_RATIO": d["LOGLEDD_RATIO"].astype(np.float64),
        })
    cat[["PLATE", "MJD", "FIBERID"]] = cat[["PLATE", "MJD", "FIBERID"]].astype(np.int64)
    m = cat.merge(prop, on=["PLATE", "MJD", "FIBERID"], how="inner")
    if len(m) < 30000:
        raise ValueError(f"DR16Q property crossmatch returned only {len(m)} rows")
    m = m[(m["LOGMBH"] > 6) & (m["LOGMBH"] < 12) &
          (m["LOGLEDD_RATIO"] > -5) & (m["LOGLEDD_RATIO"] < 2)]
    return m.set_index("OBJID")[["LOGMBH", "LOGLEDD_RATIO"]].loc[lambda x: x.index.isin(objids)]
