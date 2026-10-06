from pathlib import Path

import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parents[1]
SS = _ROOT / "temp/StandardStars"
BANDS = ["g", "r", "i"]
FID = {"g": 1, "r": 2, "i": 3}
SDSS_P3_MJD = 53500.0
MIN_EPOCHS = 10

QSO_SPECS = {"sdss": ("dr16s82_sdssLCRaw", "psMag", "psMagErr", "psMagErr_p3", "mjd"),
             "ps1": ("dr16s82_ps1LCRaw", "psfMag", "psfMagErr", "psfMagErr_p3", "obsTime"),
             "ztf": ("dr16s82_ZuberLCRaw", "mag", "magerr", "magerr_p3", "mjd")}


def excess_curves() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    curves = {}
    for sv, (f, mc, rc, pc, tc) in QSO_SPECS.items():
        d = pd.read_parquet(_ROOT / f"data/S82/{f}.parquet", columns=[mc, rc, pc, "filterID", tc])
        if sv == "sdss":
            d = d[d[tc] >= SDSS_P3_MJD]
        for b in BANDS:
            s = d[d["filterID"] == FID[b]]
            ex = np.sqrt(np.clip(s[pc].to_numpy() ** 2 - s[rc].to_numpy() ** 2, 0.0, None))
            m = s[mc].to_numpy()
            ok = np.isfinite(m) & np.isfinite(ex)
            grid = np.round(m[ok], 1)
            med = pd.Series(ex[ok]).groupby(grid).median()
            med = med[med.index.to_series().between(14.0, 26.0)]
            curves[(sv, b)] = (med.index.to_numpy(dtype=float), med.to_numpy())
            inr = ok & (m >= 14.0) & (m <= 22.0)
            chk = np.interp(m[inr][:5000], *curves[(sv, b)])
            if np.abs(chk - ex[inr][:5000]).max() > 0.005:
                raise ValueError(f"excess curve reconstruction off for {sv} {b}")
    return curves


def apply_excess(mag: np.ndarray, raw: np.ndarray, sv: str, b: str, t: np.ndarray,
                 curves: dict) -> np.ndarray:
    ex = np.interp(mag, *curves[(sv, b)])
    if sv == "sdss":
        ex = np.where(t >= SDSS_P3_MJD, ex, 0.0)
    return np.sqrt(raw ** 2 + ex ** 2)


def load_star_lcs(curves: dict) -> pd.DataFrame:
    frames = []
    d = pd.read_parquet(SS / "ss_SDSSLC_clean.parquet",
                        columns=["ssID", "mjd", "psMag", "psMagErr", "filterID"])
    for b in BANDS:
        s = d[d["filterID"] == FID[b]]
        frames.append(pd.DataFrame({
            "OBJID": "s" + s["ssID"].astype(str), "survey": "sdss", "band": b,
            "time": s["mjd"].to_numpy(), "mag": s["psMag"].to_numpy(),
            "err_raw": s["psMagErr"].to_numpy()}))
    d = pd.read_parquet(SS / "ss_PS1LC_clean.parquet",
                        columns=["ssID", "obsTime", "psfMag", "psfMagErr", "filterID"])
    for b in BANDS:
        s = d[d["filterID"] == FID[b]]
        frames.append(pd.DataFrame({
            "OBJID": "s" + s["ssID"].astype(str), "survey": "ps1", "band": b,
            "time": s["obsTime"].to_numpy(), "mag": s["psfMag"].to_numpy(),
            "err_raw": s["psfMagErr"].to_numpy()}))
    for b in BANDS:
        s = pd.read_parquet(SS / f"ss_ZuberLC_{b}_clean.parquet",
                            columns=["objectid", "mjd", "mag", "magerr"])
        frames.append(pd.DataFrame({
            "OBJID": "z" + s["objectid"].astype(str), "survey": "ztf", "band": b,
            "time": s["mjd"].to_numpy(), "mag": s["mag"].to_numpy(),
            "err_raw": s["magerr"].to_numpy()}))
    lc = pd.concat(frames, ignore_index=True)
    lc = lc[np.isfinite(lc["mag"]) & np.isfinite(lc["err_raw"]) & (lc["err_raw"] > 0)]
    parts = []
    for (sv, b), s in lc.groupby(["survey", "band"], observed=True):
        s = s.copy()
        s["err_used"] = apply_excess(s["mag"].to_numpy(), s["err_raw"].to_numpy(),
                                     sv, b, s["time"].to_numpy(), curves)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def star_stats(lc: pd.DataFrame) -> pd.DataFrame:
    g = lc.groupby(["survey", "band", "OBJID"], observed=True)
    st = g.agg(n=("mag", "size"), medmag=("mag", "median"))
    st = st[st["n"] >= MIN_EPOCHS]
    df = lc.merge(st, left_on=["survey", "band", "OBJID"], right_index=True)
    r = df["mag"] - df["medmag"]
    df["absr"] = r.abs()
    df["r2"] = r ** 2
    df["raw2"] = df["err_raw"] ** 2
    df["used2"] = df["err_used"] ** 2
    df["chi2"] = df["r2"] / df["used2"]
    g2 = df.groupby(["survey", "band", "OBJID"], observed=True)
    out = g2.agg(n=("mag", "size"), medmag=("medmag", "first"), mad=("absr", "median"),
                 r2m=("r2", "mean"), raw2=("raw2", "mean"), used2=("used2", "mean"),
                 chi2=("chi2", "mean")).reset_index()
    out["sig_rob"] = 1.4826 * out["mad"]
    out["sig_std"] = np.sqrt(out["r2m"] * out["n"] / (out["n"] - 1))
    out["chi2_dof"] = out["chi2"] * out["n"] / (out["n"] - 1)
    return out
