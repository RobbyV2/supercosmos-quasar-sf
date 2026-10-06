import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_structure_function as esf
from ensemble_structure_function import (BANDS, EDGES, CENTERS, NB, LC_PARQUET,
                                         condense_nightly, reject_outliers)
from fit_structure_function import POPDRW_COEF, POPDRW_SIG, POPDRW_LAM_OBS
from scipy.stats import norm

_ROOT = Path(__file__).resolve().parents[1]
GRID_N = 1 << 19
BATCH = 64
SEED = 20260818


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


def popdrw_params(objids: np.ndarray, z: np.ndarray, rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray]:
    cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet",
                          columns=["objectId", "PLATE", "MJD", "FIBERID"])
    cat = cat.assign(OBJID=cat["objectId"].astype(str)).set_index("OBJID").loc[objids]
    with fits.open(_ROOT / "data/dr16q_prop_May01_2024.fits", memmap=True) as h:
        d = h[1].data
        prop = pd.DataFrame({k: d[k].astype(np.int64) for k in ["PLATE", "MJD", "FIBERID"]}
                            | {k: d[k].astype(np.float64) for k in ["LOGMBH", "LOGLBOL"]})
    key = cat.reset_index()[["OBJID", "PLATE", "MJD", "FIBERID"]].astype(
        {"PLATE": np.int64, "MJD": np.int64, "FIBERID": np.int64})
    m = key.merge(prop, on=["PLATE", "MJD", "FIBERID"], how="left").set_index("OBJID")
    m = m[~m.index.duplicated()].loc[objids]
    lbol, mbh = m["LOGLBOL"].to_numpy(float), m["LOGMBH"].to_numpy(float)
    ok_l = np.isfinite(lbol) & (lbol > 43.0) & (lbol < 49.0)
    mi = 90.0 - 2.5 * np.where(ok_l, lbol, np.nan)
    mi = np.where(np.isfinite(mi), mi, np.nanmedian(mi))
    ok_m = np.isfinite(mbh) & (mbh > 6.0) & (mbh < 12.0)
    mbh = np.where(ok_m, mbh, 2.0 - 0.27 * mi)
    eps_s = rng.standard_normal(len(objids)) * POPDRW_SIG[0]
    eps_k = rng.standard_normal(len(objids)) * POPDRW_SIG[1]
    sfinf = np.zeros((len(objids), len(BANDS)))
    tau = np.zeros_like(sfinf)
    for bi, b in enumerate(BANDS):
        lam = np.log10(POPDRW_LAM_OBS[b] / (1.0 + z) / 4000.0)
        A, B, C, D = POPDRW_COEF["sfinf"]
        ls0 = A + B * lam + C * (mi + 23.0) + D * (mbh - 9.0)
        A, B, C, D = POPDRW_COEF["tau"]
        lt0 = A + B * lam + C * (mi + 23.0) + D * (mbh - 9.0)
        sdr, kdr = ls0 - 0.5 * lt0 + eps_s, lt0 + 0.5 * ls0 + eps_k
        sfinf[:, bi] = 10.0 ** (0.8 * sdr + 0.4 * kdr)
        tau[:, bi] = 10.0 ** (0.8 * kdr - 0.4 * sdr)
    print(f"popdrw draw: median SF_inf g {np.median(sfinf[:, 0]):.3f} r "
          f"{np.median(sfinf[:, 1]):.3f} mag; median tau_rest g {np.median(tau[:, 0]):.0f} "
          f"r {np.median(tau[:, 1]):.0f} d; LOGLBOL ok {int(ok_l.sum())} LOGMBH ok "
          f"{int(ok_m.sum())} of {len(objids)}")
    return sfinf, tau


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


def aggregation_recovery(n_real: int = 64, grid_n: int = 1 << 17, inputs: bool = False,
                         long_n: int = 1 << 19, shapes: bool = True,
                         on_clipped: Callable[..., None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sf_model_checks import psd_grid
    lc = pd.read_parquet(_ROOT / LC_PARQUET, filters=[("band", "in", BANDS)])
    with np.load(_ROOT / "data/ensemble_sf_covariance.npz") as f:
        selected = {b: np.sort(f[f"objids_{b}"].astype(str)) for b in BANDS}
        nbands = {b: f[f"lag_counts_{b}"].shape[1] for b in BANDS}
    cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet", columns=["objectId", "Z_DR16Q"])
    zmap = cat.assign(OBJID=cat.objectId.astype(str)).set_index("OBJID").Z_DR16Q
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
    sim_lc = nightly.rename(columns={"night": "time", "sig": "magerr"})
    base = pd.Series(0.0, index=pd.MultiIndex.from_frame(nightly[["OBJID", "band"]].drop_duplicates()))
    zero = np.zeros((len(objids), len(BANDS), NB, 3))
    sig = nightly.sig.to_numpy()
    density = nightly.groupby("OBJID").size().reindex(objids).to_numpy()
    amp_draw = np.clip(rng.normal(0, 0.2, len(objids)), -0.5, 0.5)
    tau_draw = np.clip(rng.normal(0, 0.35, len(objids)), -0.8, 0.8)
    flat = np.zeros(len(objids))
    read = lambda name, query: pd.read_csv(_ROOT / "data" / name).query(query).set_index("band").loc[list(BANDS)]
    fit = read("structure_function_object_fits.csv", "sample == 'fixed' and model == 'drw_short'")
    per = lambda d, v: np.outer(10 ** d, np.broadcast_to(v, len(BANDS)))
    spread = lambda da, dtau, a=fit.amplitude, t=fit.timescale_or_slope: (per(da, a), per(dtau, t))
    if inputs:
        regimes = [("identical_tau600", "drw", grid_n, *spread(flat, flat, 0.3, 600.0), None),
                   ("identical_tau3000", "drw", grid_n, *spread(flat, flat, 0.3, 3000.0), None),
                   ("macleod2010", "drw", grid_n, *popdrw_params(objids, z, np.random.default_rng(SEED + 28)), None)]
    else:
        regimes = [("identical", "drw", grid_n, *spread(flat, flat), None),
                   ("heterogeneous", "drw", grid_n, *spread(amp_draw, tau_draw), None),
                   ("cadence_correlated", "drw", grid_n,
                    *spread(np.sort(amp_draw)[np.argsort(np.argsort(density))], tau_draw), None)]
    if shapes and not inputs:
        bend = read("structure_function_psd_fits.csv", "sample == 'fiducial' and model == 'bending_psd'")
        start = read("structure_function_fixed_population.csv", "sample == 'fixed' and 400 < lag_center_days < 450")
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
            clipped = reject_outliers(nightly.assign(mag=noisy, sim_index=np.arange(len(nightly))))[0]
            retained = np.isin(np.arange(len(nightly)), clipped.sim_index)
            for noise, magnitudes, variance, active in [
                    ("noiseless", intrinsic, np.zeros_like(sig), np.ones(len(sig), bool)),
                    ("observed_noise", noisy, sig ** 2, np.ones(len(sig), bool)),
                    ("observed_noise_clipped", noisy, sig ** 2, retained)]:
                acc = zero.copy()
                for o, bi, positions, ii, jj, bins, _ in descriptors:
                    pi, pj = positions[ii], positions[jj]
                    use = active[pi] & active[pj]
                    nb = bins[use]
                    acc[o, bi, :, 0] = np.bincount(nb, minlength=NB)
                    acc[o, bi, :, 1] = np.bincount(nb, weights=(magnitudes[pi[use]] - magnitudes[pj[use]]) ** 2,
                                                  minlength=NB)
                    acc[o, bi, :, 2] = np.bincount(nb, weights=variance[pi[use]] + variance[pj[use]],
                                                  minlength=NB)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sf2 = esf.ensemble_second_moment(acc)
                for bi, b in enumerate(BANDS):
                    for k in range(nbands[b]):
                        case_draws.append(dict(regime=regime, psd=psd, grid_days=n, noise=noise, band=b,
                                               realization=realization, bin=k, sf2_mag2=sf2[bi, k],
                                               sf_mag=np.sqrt(max(sf2[bi, k], 0.0))))
                if noise == "observed_noise_clipped":
                    for bi, b in enumerate(BANDS):
                        keep = np.isin(objids, selected[b])
                        bounds, _ = esf.object_bootstrap(acc[keep, bi, :nbands[b]], n_boot=512,
                                                        seed=SEED + realization)
                        target = esf.ensemble_second_moment(grid_truth[keep, bi, :nbands[b]])
                        for k in range(nbands[b]):
                            coverage.append(dict(band=b, bin=k, covered=bounds[0, k] <= target[k] <= bounds[2, k],
                                                 width_mag2=bounds[2, k] - bounds[0, k]))
                    if on_clipped:
                        on_clipped(regime, realization, acc, clipped, objids, zmap)
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
            adopted = noise == "observed_noise_clipped"
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
    from fit_structure_function import _object_amplitude, _object_fit, _object_lag_kernel
    with np.load(_ROOT / "data/ensemble_sf_covariance.npz") as f:
        data = {b: (f[f"objids_{b}"].astype(str), f[f"replicates_{b}"]) for b in BANDS}
    rows, store = [], {b: [] for b in BANDS}

    def excess(regime, covariance, realization, band, a, reps, kernel):
        fits = pd.DataFrame(_object_fit(a, reps, regime, band, kernel, short_only=True)[0]).set_index("model")
        short, joint = fits.loc["drw_short"], fits.loc["drw_joint_extrapolation"]
        final = fits.loc[["drw_extrapolation"]].iloc[-1]
        rows.append(dict(regime=regime, covariance=covariance, band=band, realization=realization,
                         sf_inf_mag=short.amplitude, tau_days=short.timescale_or_slope,
                         reduced_chi2=short.reduced_chi2, final_lag_days=final.lag_center_days,
                         final_sf_mag=_object_amplitude(esf.ensemble_second_moment(a))[-1],
                         final_excess_mag=final.excess_mag, final_excess_err_mag=final.excess_err_mag,
                         final_sigma=final.excess_mag / final.excess_err_mag, joint_n_bins=int(joint.n_bins),
                         joint_chi2=joint.chi2, joint_sigma=norm.isf(joint.pvalue / 2)))

    def on_clipped(regime, realization, acc, frame, objids, zmap):
        for bi, b in enumerate(BANDS):
            ids, replicates = data[b]
            keep = np.isin(objids, ids)
            a = acc[keep, bi, :replicates.shape[1]]
            kernel = _object_lag_kernel(frame[frame.band.eq(b)], objids[keep], zmap, a.shape[1])
            np.testing.assert_array_equal(kernel[2], a[..., 0])
            boot = _object_amplitude(esf.object_bootstrap(a, n_boot=5000, seed=1927 + bi)[1])
            for covariance, reps in (("data", replicates), ("bootstrap", boot)):
                excess(regime, covariance, realization, b, a, reps, kernel)
            store[b].append((a, kernel))
            if realization == n_real - 1:
                spread = np.stack([_object_amplitude(esf.ensemble_second_moment(x)) for x, _ in store[b]])
                for i, (x, k) in enumerate(store[b]):
                    excess(regime, "realizations", i, b, x, spread, k)
                store[b].clear()

    aggregation_recovery(n_real, shapes=False, on_clipped=on_clipped)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    esf.check_flags(__file__)
    if "--excess-null" in sys.argv:
        drw_excess_null(64).to_csv(_ROOT / "data/structure_function_drw_excess_null.csv", index=False)
    elif "--aggregation-test" in sys.argv or len(sys.argv) == 1:
        output, draws = aggregation_recovery(64, 1 << 17)
        output.to_csv(_ROOT / "data/structure_function_aggregation_recovery.csv", index=False)
        draws.to_csv(_ROOT / "temp/structure_function_aggregation_draws.csv", index=False)
