import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_structure_function as esf
from ensemble_structure_function import BANDS, EDGES, CENTERS, NB, LC_PARQUET, condense_nightly
from assemble_lightcurves import redshifts

_ROOT = Path(__file__).resolve().parents[1]
GRID_N = 1 << 19
BATCH = 64
SEED = 20260818
ALL_BIN = {"drw": ("reduced_chi2_diag", "reduced_chi2_full", "final_sigma_param_scaled", "long_chi2_param_scaled",
                   "long_sigma_param_scaled"),
           "bending_psd": ("alpha", "vs_drw_delta_chi2", "vs_drw_sigma", "vs_drw_f_sigma")}


def tk95(psd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m, nf = psd.shape
    n = 2 * (nf - 1)
    amp = np.sqrt(0.25 * n * psd, dtype=np.float32)
    f = (rng.standard_normal((m, nf), dtype=np.float32)
         + 1j * rng.standard_normal((m, nf), dtype=np.float32)) * amp
    f[:, 0] = 0.0
    f[:, -1] = f[:, -1].real * np.sqrt(2.0)
    return np.fft.irfft(f, n, axis=1)


def drw_psd(freq: np.ndarray, sfinf: np.ndarray, tau: np.ndarray) -> np.ndarray:
    w = (2.0 * np.pi * freq[None, :] * tau[:, None]) ** 2
    return (2.0 * sfinf[:, None] ** 2 * tau[:, None] / (1.0 + w)).astype(np.float32)


def bending_psd(freq: np.ndarray, amp: np.ndarray, tau: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    u = 2.0 * np.pi * freq[None, 1:] * tau[:, None]
    s = np.zeros((len(tau), len(freq)), dtype=np.float32)
    s[:, 1:] = np.pi * tau[:, None] * amp[:, None] ** 2 / (u ** alpha[:, None] + u * u)
    return s


def all_bin(model: str) -> pd.DataFrame:
    return (pd.read_csv(_ROOT / "data/structure_function_all_bin_fits.csv")
            .query("sample == 'fiducial' and model == @model")
            .set_index("band", verify_integrity=True).loc[list(BANDS)])


def simulate_mags(lc: pd.DataFrame, objids: np.ndarray, z: np.ndarray, sfinf: np.ndarray,
                  tau: np.ndarray, rng: np.random.Generator, base: pd.Series,
                  grid_n: int = GRID_N, add_noise: bool = True,
                  alpha: np.ndarray | None = None) -> np.ndarray:
    oi = pd.Series(np.arange(len(objids)), index=objids)
    key = (oi.loc[lc["OBJID"]].to_numpy() * len(BANDS)
           + lc["band"].map({b: i for i, b in enumerate(BANDS)}).to_numpy())
    order = np.argsort(key, kind="stable")
    ks, first, cnt = np.unique(key[order], return_index=True, return_counts=True)
    grid = np.floor(lc["time"].to_numpy())[order].astype(np.int64) - int(lc["time"].min())
    freq = np.fft.rfftfreq(grid_n, 1.0)
    lvl = base.loc[list(zip(objids[ks // len(BANDS)],
                            [BANDS[i] for i in ks % len(BANDS)]))].to_numpy()
    out = np.empty(len(lc), dtype=np.float64)
    for a in range(0, len(ks), BATCH):
        sl = slice(a, a + BATCH)
        oo, bb = ks[sl] // len(BANDS), ks[sl] % len(BANDS)
        s, t = sfinf[oo, bb].astype(np.float32), (tau[oo, bb] * (1.0 + z[oo])).astype(np.float32)
        x = tk95(drw_psd(freq, s, t) if alpha is None else bending_psd(freq, s, t, alpha[oo, bb]), rng)
        for q, (f0, c) in enumerate(zip(first[sl], cnt[sl])):
            out[order[f0:f0 + c]] = x[q, grid[f0:f0 + c]] + lvl[a + q]
    return out + rng.normal(0.0, lc["magerr"].to_numpy()) if add_noise else out


def aggregation_recovery(n_real: int = 64, grid_n: int = 1 << 17,
                         long_n: int = 1 << 19, shapes: bool = True,
                         on_noisy: Callable[..., None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sf_model_checks import psd_grid
    lc = pd.read_parquet(_ROOT / LC_PARQUET, filters=[("band", "in", BANDS)])
    with np.load(_ROOT / "data/ensemble_sf_covariance.npz") as f:
        selected = {b: np.sort(f[f"objids_{b}"].astype(str)) for b in BANDS}
        nbands = {b: f[f"lag_counts_{b}"].shape[1] for b in BANDS}
    zmap = redshifts()
    rng = np.random.default_rng(SEED + 27)
    for b in BANDS:
        print(f"fiducial {b}: {len(selected[b])} quasars through {CENTERS[nbands[b] - 1]:.0f} rest-frame days",
              flush=True)
    lc = lc.loc[np.logical_or.reduce([(lc.band == b) & lc.OBJID.isin(selected[b]) for b in BANDS])]
    nightly = condense_nightly(lc).sort_values(["OBJID", "band", "night"]).reset_index(drop=True)
    objids = np.array(sorted(nightly.OBJID.unique()))
    z = zmap.reindex(objids).to_numpy()
    oi = {o: i for i, o in enumerate(objids)}
    descriptors = []
    for (o, b), sub in nightly.groupby(["OBJID", "band"], sort=False, observed=True):
        ii, jj = np.triu_indices(len(sub), 1)
        t = sub.night.to_numpy()
        days = np.abs(t[ii] - t[jj]).astype(np.int32)
        dt = days / (1 + zmap[o])
        bins = np.searchsorted(EDGES, dt, side="right") - 1
        keep = (bins >= 0) & (bins < nbands[b])
        descriptors.append((oi[o], BANDS.index(b), sub.index.to_numpy(),
                            ii[keep].astype(np.int32), jj[keep].astype(np.int32),
                            bins[keep].astype(np.int8), days[keep]))
    sim_lc = nightly.drop(columns="time").rename(columns={"night": "time", "sig": "magerr"})
    base = pd.Series(0.0, index=pd.MultiIndex.from_frame(nightly[["OBJID", "band"]].drop_duplicates()))
    zero = np.zeros((len(objids), len(BANDS), NB, 3))
    sig = nightly.sig.to_numpy()
    density = nightly.groupby("OBJID").size().reindex(objids).to_numpy()
    amp_draw = np.clip(rng.normal(0, 0.2, len(objids)), -0.5, 0.5)
    tau_draw = np.clip(rng.normal(0, 0.35, len(objids)), -0.8, 0.8)
    flat = np.zeros(len(objids))
    fit = all_bin("drw")
    per = lambda d, v: np.outer(10 ** d, np.broadcast_to(v, len(BANDS)))
    spread = lambda da, dtau, a=fit.sf_inf_mag, t=fit.tau_days: (per(da, a), per(dtau, t))
    regimes = [("identical", "drw", grid_n, *spread(flat, flat), None),
               ("heterogeneous", "drw", grid_n, *spread(amp_draw, tau_draw), None),
               ("cadence_correlated", "drw", grid_n,
                *spread(np.sort(amp_draw)[np.argsort(np.argsort(density))], tau_draw), None)]
    if shapes:
        bend = all_bin("bending_psd")
        start = (pd.read_csv(_ROOT / "data/structure_function_fixed_population.csv")
                 .query("sample == 'fixed' and 400 < lag_center_days < 450").set_index("band").loc[list(BANDS)])
        regimes += [("identical", psd, n, *spread(flat, flat, a, t), per(flat, alpha)) for n in (grid_n, long_n)
                    for psd, a, t, alpha in (("bending", bend.sf_inf_mag, bend.tau_days, bend.alpha),
                                             ("power_law", np.sqrt(4 / np.pi) * start.sf_mag,
                                              start.lag_center_days, 2.0))]
    interp = psd_grid()
    rows, draws = [], []
    started = time.monotonic()
    for case, (regime, psd, n, amp, tau, alpha) in enumerate(regimes):
        frequency = np.fft.rfftfreq(n, 1.0)
        truth, grid_truth = zero.copy(), zero.copy()
        for o, bi, _, ii, jj, bins, days in descriptors:
            a = None if alpha is None else alpha[o, bi]
            x = days / ((1 + z[o]) * tau[o, bi])
            shape = (-np.expm1(-x) if a is None else np.pi / 4 * x if a == 2
                     else interp(np.column_stack([np.full(x.size, a), np.log10(x)])))
            args = (frequency, amp[o:o + 1, bi], tau[o:o + 1, bi] * (1 + z[o]))
            spectrum = drw_psd(*args) if a is None else bending_psd(*args, alpha[o:o + 1, bi])
            covariance = np.fft.irfft(spectrum[0].astype(float), n)
            truth[o, bi, :, 0] = grid_truth[o, bi, :, 0] = np.bincount(bins, minlength=NB)
            truth[o, bi, :, 1] = np.bincount(bins, weights=amp[o, bi] ** 2 * shape, minlength=NB)
            grid_truth[o, bi, :, 1] = np.bincount(bins, weights=covariance[0] - covariance[days], minlength=NB)
        case_draws, coverage = [], []
        for realization in range(n_real):
            gen = np.random.default_rng(SEED + 1000 * case + realization)
            intrinsic = simulate_mags(sim_lc, objids, z, amp, tau, gen, base,
                                      grid_n=n, add_noise=False, alpha=alpha)
            noisy = intrinsic + gen.normal(0, sig)
            for noise, magnitudes, variance in [("noiseless", intrinsic, np.zeros_like(sig)),
                                                ("observed_noise", noisy, sig ** 2)]:
                acc = zero.copy()
                for o, bi, positions, ii, jj, bins, _ in descriptors:
                    pi, pj = positions[ii], positions[jj]
                    acc[o, bi, :, 0] = np.bincount(bins, minlength=NB)
                    acc[o, bi, :, 1] = np.bincount(bins, weights=(magnitudes[pi] - magnitudes[pj]) ** 2, minlength=NB)
                    acc[o, bi, :, 2] = np.bincount(bins, weights=variance[pi] + variance[pj], minlength=NB)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sf2 = esf.ensemble_second_moment(acc)
                for bi, b in enumerate(BANDS):
                    for k in range(nbands[b]):
                        case_draws.append(dict(regime=regime, psd=psd, grid_days=n, noise=noise, band=b,
                                               realization=realization, bin=k, sf2_mag2=sf2[bi, k],
                                               sf_mag=np.sqrt(max(sf2[bi, k], 0.0))))
                if noise == "observed_noise":
                    for bi, b in enumerate(BANDS):
                        keep = np.isin(objids, selected[b])
                        bounds, _ = esf.object_bootstrap(acc[keep, bi, :nbands[b]], n_boot=512,
                                                        seed=SEED + realization)
                        target = esf.ensemble_second_moment(grid_truth[keep, bi, :nbands[b]])
                        for k in range(nbands[b]):
                            coverage.append(dict(band=b, bin=k, covered=bounds[0, k] <= target[k] <= bounds[2, k],
                                                 width_mag2=bounds[2, k] - bounds[0, k]))
                    if on_noisy:
                        on_noisy(regime, realization, acc, objids)
            if realization == 0 or (realization + 1) % 8 == 0:
                print(f"aggregation {regime} {psd} grid {n} {realization + 1}/{n_real}, "
                      f"elapsed {time.monotonic() - started:.1f} s", flush=True)
        draws += case_draws
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            target_sf = np.sqrt(esf.ensemble_second_moment(truth))
            grid_sf = np.sqrt(esf.ensemble_second_moment(grid_truth))
        cov = pd.DataFrame(coverage)
        for (noise, b, k), group in pd.DataFrame(case_draws).groupby(["noise", "band", "bin"]):
            bi = BANDS.index(b)
            value, squared = group.sf_mag.to_numpy(), group.sf2_mag2.to_numpy()
            target, gtarget = target_sf[bi, k], grid_sf[bi, k]
            c = cov[(cov.band == b) & (cov.bin == k)]
            adopted = noise == "observed_noise"
            rows.append(dict(regime=regime, psd=psd, grid_days=n, noise=noise, band=b, sample="fiducial",
                             lag_center_days=CENTERS[k], n_objects=len(selected[b]), n_real=n_real,
                             n_pairs=int(truth[:, bi, k, 0].sum()), sf_mag=value.mean(),
                             sf_real_sd_mag=value.std(ddof=1), sf_mc_se_mag=value.std(ddof=1) / np.sqrt(n_real),
                             sf2_mag2=squared.mean(), sf2_mc_se_mag2=squared.std(ddof=1) / np.sqrt(n_real),
                             analytic_target_mag=target, grid_target_mag=gtarget,
                             bias_mag=value.mean() - target, grid_bias_mag=value.mean() - gtarget,
                             nonpositive_real_fraction=(squared <= 0).mean(),
                             bootstrap_coverage=c.covered.mean() if adopted else np.nan,
                             bootstrap_width_mag2=c.width_mag2.mean() if adopted else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(draws)


def drw_excess_null(n_real: int = 64) -> pd.DataFrame:
    from sf_model_checks import adopted_reps, all_bin_fits, psd_grid
    grid = psd_grid()
    cols = [f"{m}_{k}" for m, keys in ALL_BIN.items() for k in keys]
    measured = pd.concat([all_bin(m)[list(keys)].add_prefix(f"{m}_") for m, keys in ALL_BIN.items()], axis=1)
    with np.load(_ROOT / "data/ensemble_sf_covariance.npz") as f:
        fiducial = {b: (f[f"objids_{b}"].astype(str), f[f"lag_counts_{b}"].shape[1]) for b in BANDS}
    rows = []
    started = time.monotonic()

    def on_noisy(regime, realization, acc, objids):
        new = []
        for bi, b in enumerate(BANDS):
            ids, nb = fiducial[b]
            a = acc[np.isin(objids, ids), bi, :nb]
            fits = pd.DataFrame(all_bin_fits(a, grid, adopted_reps(b, "fiducial", a))[0]).set_index("model")
            new.append(dict(regime=regime, band=b, realization=realization,
                            **{f"{m}_{k}": fits.at[m, k] for m, keys in ALL_BIN.items() for k in keys}))
        rows.extend(new)
        print(f"excess null {regime} {realization + 1}/{n_real}, DRW long sigma "
              + " ".join(f"{r['band']} {r['drw_long_sigma_param_scaled']:.2f}" for r in new)
              + f", elapsed {time.monotonic() - started:.0f} s", flush=True)

    aggregation_recovery(n_real, shapes=False, on_noisy=on_noisy)
    d = pd.DataFrame(rows)
    by = d.groupby(["regime", "band"], sort=False)[cols]
    hit = d[cols].ge(measured.loc[d.band].to_numpy())
    both = hit.groupby([d.regime, d.realization], sort=False).all().groupby(level=0, sort=False).sum()
    k = pd.concat([hit.groupby([d.regime, d.band], sort=False).sum(),
                   both.assign(band="joint").set_index("band", append=True)])
    mean = by.mean()
    summary = pd.concat({"mean": mean, "sd": by.std(), "median": by.median(), "p95": by.quantile(0.95),
                         "max": by.max(), "measured": measured.loc[mean.index.get_level_values("band")].set_axis(mean.index),
                         "n_at_or_above": k, "p_empirical": (k + 1) / (n_real + 1)}, names=["summary"])
    out = pd.concat([d, summary.reset_index()], ignore_index=True)
    return out.astype({"realization": "Int64"})[["regime", "band", "realization", "summary", *cols]]


if __name__ == "__main__":
    esf.check_flags(__file__)
    if "--excess-null" in sys.argv:
        drw_excess_null(64).to_csv(_ROOT / "data/structure_function_drw_excess_null.csv", index=False)
    elif "--aggregation-test" in sys.argv:
        output, draws = aggregation_recovery(64, 1 << 17)
        output.to_csv(_ROOT / "data/structure_function_aggregation_recovery.csv", index=False)
        draws.to_csv(_ROOT / "temp/structure_function_aggregation_draws.csv", index=False)
    else:
        raise SystemExit("usage: --aggregation-test | --excess-null")
