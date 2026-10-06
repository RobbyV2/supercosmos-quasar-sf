from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist

import ensemble_structure_function as esf
from ensemble_structure_function import (BANDS, CENTERS, EDGES, NB, accumulate,
                                         ledoit_wolf)

_ROOT = Path(__file__).resolve().parent.parent
SHORT_MAX_D = 3000.0
PLATEAU_MIN_D = 400.0  # plateau fit start (CCD lags 422-2371 d)
LONG_MIN_D = 11 * 365.25  # 11 rest-frame yr; excess bins beyond
DRW_BOUNDS = ([1e-3, 1.0], [3.0, 1e6])
STONE_FITS = "data/stone2022_ensemble_sf_psd.fits.gz"
STONE_LC = "data/stone2022_lightcurves.fits.gz"
STONE_FLOOR_END_D = {"g": 5.0, "r": 20.0, "i": 40.0}
FIXED_CSV = "data/structure_function_fixed_population.csv"
GLS_TAU0 = [50.0, 200.0, 700.0, 2000.0]


def stone_grid():
    t = Table.read(_ROOT / STONE_FITS)
    row = t[[str(s).strip() == "Total" for s in t["Subsample"]]][0]
    grid = np.asarray(row["DT_REST_g"], float)
    lgrid = np.log10(grid)
    lstep = float(np.median(np.diff(lgrid)))
    edges = 10.0 ** np.concatenate([lgrid - lstep / 2, [lgrid[-1] + lstep / 2]])
    return grid, edges


def plateau_bins(dt, lo=PLATEAU_MIN_D, hi=SHORT_MAX_D):
    return (dt >= lo) & (dt <= hi)


def drw(dt, sf_inf, tau):
    return sf_inf * np.sqrt(1.0 - np.exp(-dt / tau))


def powerlaw(dt, a, gamma):
    return a * (dt / 1000.0) ** gamma


def fit(model, dt, sf, err, p0, bounds=(-np.inf, np.inf)):
    popt, pcov = curve_fit(model, dt, sf, p0=p0, sigma=err, absolute_sigma=True,
                           bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    r = sf - model(dt, *popt)
    chi2 = float(r @ np.linalg.solve(err, r)) if np.ndim(err) == 2 else np.sum((r / err) ** 2)
    dof = len(dt) - len(popt)
    return popt, perr, pcov, chi2 / dof


MACLEOD_POSS1 = dict(lag_center_days=7943.732775527159, sf_mag=0.3498202596480719, sf_lo_mag=0.2999244789127741,
                     sf_hi_mag=0.3997951486030456, wavelength_lo_angstrom=2000, wavelength_hi_angstrom=3000)  # MacLeod+2012 Fig. 17 SDSS-POSS-I


def literature_choices(n_boot=2000):
    ours = pd.read_csv(_ROOT / FIXED_CSV)
    mac = pd.Series(MACLEOD_POSS1)
    s = pd.read_parquet(_ROOT / "data/plate_completeness_sample.parquet")
    with np.load(_ROOT / "data/structure_function_sampling_covariance.npz") as f:
        ids = {b: f[f"fiducial_ids_{b}"] for b in BANDS}
    z = {b: s[s.band.eq(b)].set_index("OBJID").z.reindex(ids[b]).to_numpy() for b in BANDS}
    t = Table.read(_ROOT / STONE_FITS)
    row = t[[str(x).strip() == "Total" for x in t["Subsample"]]][0]
    grid, _ = stone_grid()
    lgrid = np.log10(grid)
    with fits.open(_ROOT / STONE_LC) as h:
        per = {b: [np.array([r[f"{c}_{b}"] for r in h[1].data], float) for c in ("SF", "DT_REST")] for b in BANDS}
        stone_coords = SkyCoord(h[1].data["RA"] * u.deg, h[1].data["DEC"] * u.deg)
    rng = np.random.default_rng(2026)
    rows = []
    for b in BANDS:
        sf, dt = per[b]
        kid = np.rint((np.log10(dt) - lgrid[0]) / np.median(np.diff(lgrid))).astype(int)
        pub, err = (np.asarray(row[c], float) for c in (f"SF_{b}", f"SF_{b}_ERR"))
        for k in np.flatnonzero(np.isfinite(pub) & (grid >= STONE_FLOOR_END_D[b]) & (pub > 0)):
            v = sf[(kid == k) & np.isfinite(sf)]
            boot = np.percentile(np.median(rng.choice(v, (n_boot, v.size)), 1), [16, 84])
            rows.append(dict(kind="stone_error", band=b, lag_days=grid[k], n_objects=v.size, sf_mag=pub[k],
                             sf_err_mag=err[k], std_over_n_mag=v.std() / v.size,
                             std_over_sqrt_n_mag=v.std() / np.sqrt(v.size), half_width_mag=np.diff(boot)[0] / 2))
        tab = ours[ours.band.eq(b)].sort_values("lag_center_days")
        for src, lag, lit, hw in [("stone", grid[k], pub[k], err[k]), ("macleod_poss1", mac.lag_center_days,
                                  mac.sf_mag, (mac.sf_hi_mag - mac.sf_lo_mag) / 2)]:
            o = tab.iloc[np.searchsorted(EDGES, lag) - 1]
            rows.append(dict(kind="matched_bin", source=src, band=b, lag_days=lag, sf_mag=lit, sf_err_mag=hw,
                             our_lag_center_days=o.lag_center_days, our_lag_lo_days=o.lag_lo_days,
                             our_lag_hi_days=o.lag_hi_days, our_sf_mag=o.sf_mag,
                             our_half_width_mag=(o.sf_hi_mag - o.sf_lo_mag) / 2, n_objects=o.n_objects))
        lam = POPDRW_LAM_OBS[b] / (1 + z[b])
        rows.append(dict(kind="rest_wavelength", band=b, n_objects=lam.size, lam_obs_angstrom=POPDRW_LAM_OBS[b],
                         lam_rest_median_angstrom=np.median(lam), lam_rest_q16_angstrom=np.percentile(lam, 16),
                         lam_rest_q84_angstrom=np.percentile(lam, 84), frac_2000_3000=np.mean(
                             (lam >= mac.wavelength_lo_angstrom) & (lam <= mac.wavelength_hi_angstrom))))
    cat = pd.read_parquet(_ROOT / "data/S82/Catalog.parquet", columns=["objectId", "RA", "DEC"])
    match, separation, _ = stone_coords.match_to_catalog_sky(
        SkyCoord(cat.RA.to_numpy() * u.deg, cat.DEC.to_numpy() * u.deg))
    matched = separation.arcsec < 1.0
    stone_ids = cat.iloc[match[matched]].objectId.astype(str).to_numpy()
    prop = esf.load_properties(np.unique(np.concatenate([*ids.values(), stone_ids])))
    samples = [("fiducial", b, ids[b], len(ids[b])) for b in BANDS]
    samples.append(("stone", "g,r", stone_ids, len(stone_coords)))
    for source, band, members, total in samples:
        p = prop.reindex(members).dropna()
        rows.append(dict(kind="sample_properties", source=source, band=band, n_objects=total,
                         n_matched=len(members), n_valid_properties=len(p),
                         median_log_mbh=p.LOGMBH.median(), median_log_edd=p.LOGLEDD_RATIO.median(),
                         median_edd=np.median(10 ** p.LOGLEDD_RATIO),
                         match_radius_arcsec=1.0 if source == "stone" else np.nan,
                         max_match_separation_arcsec=separation.arcsec[matched].max() if source == "stone" else np.nan))
    pd.DataFrame(rows).to_csv(_ROOT / "data/structure_function_literature_choices.csv", index=False)


POPDRW_COEF = {"sfinf": (-0.51, -0.479, 0.131, 0.18), "tau": (2.4, 0.17, 0.03, 0.21)}
POPDRW_SIG = (0.149, 0.26)
POPDRW_LAM_OBS = {"g": 4800.0, "r": 6250.0}


def _pw_quant(v, w, ps=(0.1, 0.5, 0.9)):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return np.interp(ps, cw / cw[-1], v[o]) if cw[-1] > 0 else np.full(len(ps), np.nan)


def _cell_index(z, lum, nq):
    c = pd.qcut(z, nq, labels=False) * nq + pd.qcut(lum, nq, labels=False)
    return c.astype(int), nq * nq


def _object_amplitude(v):
    return np.sqrt(np.maximum(v, 0.0))


def _object_summary(a, z, lum, n_boot=5000, seed=31, weights=None):
    v = esf.ensemble_second_moment(a, weights=weights)
    bounds, reps = esf.object_bootstrap(a, n_boot=n_boot, seed=seed, weights=weights)
    sf, limits, samples = (_object_amplitude(x) for x in (v, bounds, reps))
    n = (a[..., 0] > 0).sum(axis=0)
    rows = []
    for k in range(a.shape[1]):
        use = a[:, k, 0] > 0
        w = np.ones(len(a)) if weights is None else np.broadcast_to(weights, a.shape[:-1])[:, k]
        use &= w > 0
        zq = _pw_quant(z[use], w[use]) if use.any() else np.full(3, np.nan)
        lq = _pw_quant(lum[use], w[use]) if use.any() else np.full(3, np.nan)
        rows.append(dict(lag_center_days=CENTERS[k], lag_lo_days=EDGES[k], lag_hi_days=EDGES[k+1],
                         sf2_mag2=v[k], sf_mag=sf[k], sf_lo_mag=limits[0, k], sf_hi_mag=limits[2, k],
                         sf_err_mag=.5*(limits[2, k]-limits[0, k]),
                         n_objects=int(n[k]), n_pairs=int(a[:, k, 0].sum()),
                         n_negative_objects=int((esf.object_second_moments(a)[:, k] < 0).sum()),
                         z_q10=zq[0], z_med=zq[1], z_q90=zq[2],
                         logl_q10=lq[0], logl_med=lq[1], logl_q90=lq[2]))
    return pd.DataFrame(rows), samples


def _object_lag_kernel(nightly, ids, zmap, n_bins, subdivisions=32):
    nodes = np.zeros((len(ids), n_bins, subdivisions))
    weights = np.zeros_like(nodes)
    count = np.zeros((len(ids), n_bins), dtype=int)
    edges = np.geomspace(EDGES[0], EDGES[n_bins], n_bins*subdivisions+1)
    groups = dict(tuple(nightly.groupby('OBJID', sort=False)))
    for q, oid in enumerate(ids):
        t = groups[oid].night.to_numpy(float)
        ii, jj = np.triu_indices(len(t), 1)
        lag = abs(t[jj]-t[ii])/(1+zmap[oid])
        h = np.searchsorted(edges, lag, side='right')-1
        good = (h >= 0) & (h < n_bins*subdivisions)
        nh = np.bincount(h[good], minlength=n_bins*subdivisions).reshape(n_bins, subdivisions)
        total = np.bincount(h[good], weights=lag[good], minlength=n_bins*subdivisions).reshape(n_bins, subdivisions)
        nodes[q] = np.divide(total, nh, out=np.zeros_like(total), where=nh>0)
        count[q] = nh.sum(axis=1)
        weights[q] = np.divide(nh, count[q,:,None], out=np.zeros_like(total), where=count[q,:,None]>0)
    return nodes, weights, count


def _object_sampled_models(kernel):
    nodes, weights = kernel[:2]
    def mean(values):
        return (values*weights).sum(axis=2).mean(axis=0)
    def choose(x):
        return np.abs(np.asarray(x)[:,None]-CENTERS[:nodes.shape[1]]).argmin(axis=1)
    def sampled_drw(x, amplitude, tau):
        return amplitude*np.sqrt(mean(-np.expm1(-nodes/tau)))[choose(x)]
    def sampled_powerlaw(x, amplitude, slope):
        # empty cells have zero weight; floor avoids 0**negative
        v = (np.maximum(nodes, 1e-9)/1000.)**(2*slope)
        return amplitude*np.sqrt(mean(v))[choose(x)]
    def drw_jacobian(x, amplitude, tau):
        root = np.sqrt(mean(-np.expm1(-nodes/tau)))
        derivative = -amplitude*mean(np.exp(-nodes/tau)*nodes)/(2*tau**2*root)
        return np.column_stack([root, derivative])[choose(x)]
    return sampled_drw, sampled_powerlaw, drw_jacobian


def _extrapolation_cov(cov, fitted, jac, pcov, rchi2=1.):
    cross = cov[:, fitted] @ np.linalg.solve(cov[np.ix_(fitted, fitted)], jac[fitted]) @ pcov @ jac.T
    return cov + max(rchi2, 1.) * jac @ pcov @ jac.T - cross - cross.T


def _object_fit(a, reps, name, band, kernel=None, fit_range=(PLATEAU_MIN_D, SHORT_MAX_D), short_only=False):
    v = esf.ensemble_second_moment(a)
    sf = _object_amplitude(v)
    ok = (v > 0) & np.isfinite(reps).all(axis=0)
    short = ok & plateau_bins(CENTERS[:len(v)], *fit_range)
    if short.sum() < 4:
        return None
    cov = np.cov(reps[:, ok], rowvar=False)
    sd = np.sqrt(np.diag(cov))
    corr, _, _ = ledoit_wolf(reps[:, ok]/sd)
    cov = corr*np.outer(sd, sd)
    short_ok = short[ok]
    dt, y = CENTERS[:len(v)][ok], sf[ok]
    if kernel is None:
        raise ValueError('Object-first model fits require the actual pair-lag kernel')
    sampled_drw, sampled_powerlaw, sampled_jacobian = _object_sampled_models(kernel)
    results = []
    for model, key, m in [(sampled_drw, 'drw_short', short_ok), (sampled_drw, 'drw_all', np.ones(len(y), bool)),
                          (sampled_powerlaw, 'powerlaw_all', np.ones(len(y), bool))]:
        if short_only and key != 'drw_short':
            continue
        try:
            if model == sampled_drw:
                trials = [fit(model, dt[m], y[m], cov[np.ix_(m, m)], [y[m].max(), t], DRW_BOUNDS)
                          for t in GLS_TAU0]
                p, e, pc, rc = min(trials, key=lambda r:r[3])
            else:
                p, e, pc, rc = fit(model, dt[m], y[m], cov[np.ix_(m, m)], [y[m].max(), .25])
            results.append(dict(sample=name, band=band, model=key, amplitude=p[0],
                                amplitude_err=e[0], timescale_or_slope=p[1],
                                timescale_or_slope_err=e[1], reduced_chi2=rc,
                                n_bins=int(m.sum()), n_objects=len(a)))
            if key == 'drw_short':
                short_p, short_pc, scale = p, pc, max(rc, 1.)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            results.append(dict(sample=name, band=band, model=key, n_bins=int(m.sum()), n_objects=len(a)))
    if 'short_p' not in locals():
        return None
    excess = y-sampled_drw(dt, *short_p)
    residual_cov = _extrapolation_cov(cov, short_ok, sampled_jacobian(dt, *short_p), short_pc, scale)
    last = np.flatnonzero(dt > LONG_MIN_D)
    rise = []
    k0 = int(np.argmin(abs(CENTERS[:len(v)]-2371)))
    for k in np.flatnonzero(CENTERS[:len(v)] > LONG_MIN_D):
        diff = reps[:, k]-reps[:, k0]
        q = np.percentile(diff, [16, 50, 84])
        rise.append(dict(sample=name, band=band, lag_center_days=CENTERS[k],
                         sf_anchor_mag=sf[k0], sf_mag=sf[k], rise_mag=sf[k]-sf[k0],
                         rise_lo_mag=q[0], rise_hi_mag=q[2],
                         rise_bootstrap_sigma=float(diff.std(ddof=1)),
                         ratio=sf[k]/sf[k0]))
    for j in last:
        results.append(dict(sample=name, band=band, model='drw_extrapolation',
                            lag_center_days=dt[j], excess_mag=excess[j],
                            excess_err_mag=np.sqrt(max(residual_cov[j,j],0)), scale_factor=scale, n_objects=len(a)))
    if len(last):
        r=excess[last]
        c=residual_cov[np.ix_(last,last)]
        x=float(r@np.linalg.solve(c,r))
        results.append(dict(sample=name,band=band,model='drw_joint_extrapolation',
                            chi2=x,n_bins=len(last),pvalue=chi2_dist.sf(x,len(last)),scale_factor=scale,n_objects=len(a)))
    return results, rise, short_p, cov, scale*short_pc


def _object_plate_jackknife(nightly, ids, zmap, a, band):
    table = {o: k for k, o in enumerate(ids)}
    plates = np.sort(nightly.loc[nightly.sss, 'plate'].unique())
    plate_index = {p:i for i,p in enumerate(plates)}
    base = esf.object_second_moments(a)
    numer = np.nansum(base, axis=0)[None, :].repeat(len(plates), axis=0)
    denom = np.isfinite(base).sum(axis=0)[None, :].repeat(len(plates), axis=0)
    contributing = np.zeros((len(plates), a.shape[1]), bool)
    for o, sub in nightly.groupby('OBJID', sort=False):
        iq = table[o]
        t, m, e = (sub[k].to_numpy(float) for k in ['night','mag','sig'])
        pl, pc = sub.plate.to_numpy(), sub.pcode.to_numpy()
        fv = sub[esf.FV_COLS].to_numpy(float)
        for p in np.unique(pl[sub.sss.to_numpy(bool)]):
            ip = plate_index[p]
            rows = np.flatnonzero(pl == p)
            ii, jj = np.unique(np.sort(np.stack([np.repeat(rows,len(sub)), np.tile(np.arange(len(sub)),len(rows))]),axis=0),axis=1)
            valid = ii != jj
            ii, jj = ii[valid], jj[valid]
            bins = np.searchsorted(EDGES, abs(t[jj]-t[ii])/(1+zmap[o]), side='right')-1
            valid = (bins >= 0) & (bins < a.shape[1])
            ii,jj,bins = ii[valid],jj[valid],bins[valid]
            n = np.bincount(bins,minlength=a.shape[1])
            v = np.bincount(bins,weights=(m[jj]-m[ii])**2-e[ii]**2-e[jj]**2-fv[ii,pc[jj]],minlength=a.shape[1])
            left = a[iq,:,0]-n
            rem = np.divide(a[iq,:,1]-a[iq,:,2]-v,left,out=np.full(len(left),np.nan),where=left>0)
            numer[ip] += np.nan_to_num(rem)-np.nan_to_num(base[iq])
            denom[ip] += np.isfinite(rem).astype(int)-np.isfinite(base[iq]).astype(int)
            contributing[ip] |= n>0
    theta = _object_amplitude(np.divide(numer,denom,out=np.full_like(numer,np.nan),where=denom>0))
    err = np.full(a.shape[1], np.nan)
    for k in range(a.shape[1]):
        t = theta[contributing[:,k],k]
        if len(t)>1:
            err[k] = np.sqrt((len(t)-1)/len(t)*np.sum((t-t.mean())**2))
    return err, contributing.sum(axis=0), theta, contributing


def _object_completeness_sweep(acc, ids, sample, z, lum, outdir, n_boot=10000):
    from plate_completeness import ADOPTED_SELECTION
    rows, compare, panels, matched_rows, shape_rows, criteria = [], [], {}, [], [], []
    candidates = [f'{kind}_cut_{margin}' for kind in ['plate','all'] for margin in ['025','050','075','100','125']]
    adopted = {}
    for bi, band in enumerate(BANDS):
        s = sample[sample.band.eq(band)].set_index('OBJID').reindex(ids)
        a = acc[:,bi]
        endpoint = 12 if band == 'g' else 13
        covered = (a[:,:endpoint+1,0]>0).all(axis=1)
        masks = np.stack([s[c].fillna(False).to_numpy(bool)&covered for c in candidates],axis=1)
        expanded = np.broadcast_to(a[:,None,:endpoint+1],(len(a),len(candidates),endpoint+1,6))
        w = np.broadcast_to(masks[:,:,None],expanded.shape[:-1]).astype(float)
        point = esf.ensemble_second_moment(expanded, weights=w)
        bounds, reps = esf.object_bootstrap(expanded,n_boot=n_boot,seed=927+bi,weights=w)
        pointsf, bounds, reps = (_object_amplitude(x) for x in (point,bounds,reps))
        long = np.flatnonzero(CENTERS[:endpoint+1] >= 2371)
        for c, name in enumerate(candidates):
            use = masks[:,c]
            for k in range(endpoint+1):
                rows.append(dict(band=band,selection=name,lag_center_days=CENTERS[k],n_fixed=int(use.sum()),
                                 sf_mag=pointsf[c,k],sf_lo_mag=bounds[0,c,k],sf_hi_mag=bounds[2,c,k],
                                 median_z=float(np.nanmedian(z[use])),median_loglbol=float(np.nanmedian(lum[use]))))
        for ci,name in enumerate(candidates):
            for cj in range(ci+1,len(candidates)):
                if candidates[cj].split('_')[0] != name.split('_')[0]:
                    continue
                delta = reps[:,ci]-reps[:,cj]
                for k in long:
                    q = np.percentile(delta[:,k],[16,84])
                    compare.append(dict(band=band,selection=name,comparison=candidates[cj],lag_center_days=CENTERS[k],
                                        difference_mag=pointsf[ci,k]-pointsf[cj,k],difference_lo_mag=q[0],
                                        difference_hi_mag=q[1],difference_se_mag=delta[:,k].std(ddof=1),
                                        n_fixed=int(masks[:,ci].sum()),n_comparison=int(masks[:,cj].sum()),bootstrap_draws=n_boot))
        full=masks[:,5]
        ze=np.quantile(z[full],[0,.5,1])
        le=np.quantile(lum[full],[0,.5,1])
        cell=np.clip(np.digitize(z,ze[1:-1]),0,1)*2+np.clip(np.digitize(lum,le[1:-1]),0,1)
        occupancy=np.stack([np.bincount(cell[masks[:,c]],minlength=4) for c in range(5,10)])
        target=(occupancy.min(axis=0)>=5).astype(float)
        if target.any():
            wc=np.stack([np.divide(target,n,out=np.zeros(4),where=n>0)[cell] for n in occupancy],axis=1)
            mw=w[:,5:]*wc[:,:,None]
            mpoint=_object_amplitude(esf.ensemble_second_moment(expanded[:,5:],weights=mw))
            mbounds,mreps=esf.object_bootstrap(expanded[:,5:],n_boot=n_boot,seed=927+bi,weights=mw)
            mbounds=_object_amplitude(mbounds)
            for c,name in enumerate(candidates[5:]):
                for k in long:
                    matched_rows.append(dict(band=band,selection=name,lag_center_days=CENTERS[k],
                                             sf_mag=mpoint[c,k],sf_lo_mag=mbounds[0,c,k],sf_hi_mag=mbounds[2,c,k],
                                             n_objects=int((mw[:,c,k]>0).sum()),n_shared_cells=int(target.sum())))
        anchor=int(np.argmin(abs(CENTERS[:endpoint+1]-2371)))
        normalized=pointsf/pointsf[:,anchor,None]
        nreps=reps/reps[:,:,anchor,None]
        for ci in range(5,9):
            for cj in range(ci+1,10):
                delta=nreps[:,ci]-nreps[:,cj]
                for k in long:
                    q=np.percentile(delta[:,k],[16,84])
                    shape_rows.append(dict(band=band,selection=candidates[ci],comparison=candidates[cj],
                                           lag_center_days=CENTERS[k],ratio_difference=normalized[ci,k]-normalized[cj,k],
                                           difference_lo=q[0],difference_hi=q[1],difference_se=delta[:,k].std(ddof=1)))
        checks=[]
        for ci in range(5,8):
            all_stable=True
            for cj in range(ci+1,10):
                delta=reps[:,ci,long]-reps[:,cj,long]
                se=delta.std(axis=0,ddof=1)
                centered=(delta-delta.mean(axis=0))/se
                threshold=np.percentile(np.max(abs(centered),axis=1),95)
                statistic=float(np.max(abs(pointsf[ci,long]-pointsf[cj,long])/se))
                all_stable &= statistic <= threshold and masks[:,cj].sum()>=50
                criteria.append(dict(band=band,selection=candidates[ci],comparison=candidates[cj],
                                     maximum_standardized_difference=statistic,threshold95=threshold,
                                     simultaneous_p=float((np.max(abs(centered),axis=1)>=statistic).mean()),
                                     n_comparison=int(masks[:,cj].sum()),bootstrap_draws=n_boot))
            eligible = s[candidates[ci]].fillna(False).to_numpy(bool)
            recovery_table=pd.read_csv(_ROOT/'data/plate_completeness_cut_sweep.csv')
            recovery_row=recovery_table[recovery_table.band.eq(band)&recovery_table.selection.eq(candidates[ci])].iloc[0]
            recovery=float(recovery_row.n_objects_archive_detection/recovery_row.n_parent)
            checks.append((candidates[ci], all_stable, recovery,int(masks[:,ci].sum())))
        pd.DataFrame(rows).to_csv(outdir/'structure_function_completeness_sweep.csv',index=False)
        pd.DataFrame(compare).to_csv(outdir/'structure_function_completeness_differences.csv',index=False)
        pd.DataFrame(matched_rows).to_csv(outdir/'structure_function_completeness_matched.csv',index=False)
        pd.DataFrame(shape_rows).to_csv(outdir/'structure_function_completeness_shapes.csv',index=False)
        pd.DataFrame(criteria).to_csv(outdir/'structure_function_completeness_stability.csv',index=False)
        adopted[band]=ADOPTED_SELECTION[band]
        print(f'{band}: completeness candidates {checks}; adopted {adopted[band]}',flush=True)
        panels[band]=(pointsf,bounds,candidates,endpoint)
    pd.DataFrame(rows).to_csv(outdir/'structure_function_completeness_sweep.csv',index=False)
    pd.DataFrame(compare).to_csv(outdir/'structure_function_completeness_differences.csv',index=False)
    return adopted


def object_first_results():
    import hashlib
    import inspect
    import json
    import shutil
    from plate_completeness import load_completeness_lightcurves

    output=esf.CACHE/'object_first'
    output.mkdir(parents=True,exist_ok=True)
    sample=pd.read_parquet(_ROOT/'data/plate_completeness_sample.parquet')
    sample['OBJID']=sample.OBJID.astype(str)
    ids=np.sort(sample.loc[sample.plate_cut_025,'OBJID'].unique()).astype(str)
    cats=sample[sample.band.eq('g')].set_index('OBJID').reindex(ids)
    z,lum=cats.z.to_numpy(float),cats.loglbol.to_numpy(float)
    zmap=pd.Series(z,index=ids)
    sources=['data/detection_correction_components.parquet','data/plate_native_pair_variance.csv']
    kernels=[accumulate,esf.clean_nightly,esf.condense_nightly,esf.reject_outliers,esf.floor_variance,
             load_completeness_lightcurves]
    stamp=(_ROOT/esf.LC_PARQUET).stat()
    code=''.join(inspect.getsource(f) for f in kernels)+f'{stamp.st_size}:{stamp.st_mtime_ns}'
    signature=hashlib.sha256(b''.join((_ROOT/p).read_bytes() for p in sources)+ids.astype('U').tobytes()+code.encode()).hexdigest()
    cache=output/'object_accumulator.npz'
    completed=0
    acc=np.zeros((len(ids),len(BANDS),NB,6))
    if cache.exists():
        with np.load(cache) as f:
            if str(f['signature'])==signature:
                acc=f['acc']
                completed=int(f['completed']) if 'completed' in f else len(ids)
    nightly=None
    if completed<len(ids):
        lc=load_completeness_lightcurves(ids)
        print(f'Accumulating {len(ids)} quasars, {len(lc)} epochs without detection corrections',flush=True)
        nightly=esf.clean_nightly(lc)[0]
        del lc
        for start in range(completed,len(ids),1000):
            stop=min(start+1000,len(ids))
            part=ids[start:stop]
            acc[start:stop]=accumulate(nightly[nightly.OBJID.isin(part)],part,zmap,resid=True)
            np.savez_compressed(cache,acc=acc,ids=ids,signature=signature,completed=stop)
            print(f'Accumulated {stop}/{len(ids)} objects',flush=True)
    adopted=_object_completeness_sweep(acc,ids,sample,z,lum,output)
    for band in BANDS:
        sample.loc[sample.band.eq(band),'adopted_keep']=sample.loc[sample.band.eq(band),adopted[band]]
    sample.to_parquet(_ROOT/'data/plate_completeness_sample.parquet',index=False)
    rows,comparisons,fits_rows,rises=[],[],[],[]
    bootstrap_outputs={}
    selected={}
    for bi,band in enumerate(BANDS):
        s=sample[sample.band.eq(band)].set_index('OBJID').reindex(ids)
        available=s[adopted[band]].fillna(False).to_numpy(bool)
        endpoint=12 if band=='g' else 13
        keep=available&(acc[:,bi,:endpoint+1,0]>0).all(axis=1)
        a=acc[keep,bi,:endpoint+1]
        oid=ids[keep]
        table,reps=_object_summary(a,z[keep],lum[keep],seed=1927+bi)
        table['sample']='fixed'
        table['band']=band
        table['selection']=adopted[band]
        if nightly is None:
            nt=esf.clean_nightly(load_completeness_lightcurves(oid).query('band == @band'))[0]
        else:
            nt=nightly[nightly.OBJID.isin(oid)&nightly.band.eq(band)]
        jack,npl,theta,contributing=_object_plate_jackknife(nt,oid,zmap,a,band)
        table['sf_jack_err_mag']=jack
        table['n_plates_jack']=npl
        table['sf_adopt_err_mag']=table.sf_err_mag
        rows.append(table)
        kernel=_object_lag_kernel(nt,oid,zmap,a.shape[1])
        np.testing.assert_array_equal(kernel[2],a[...,0])
        analysis=_object_fit(a,reps,'fixed',band,kernel)
        if analysis is None:
            raise ValueError(f'{band}: fixed sample has insufficient bins for DRW fit')
        fr,rr,_,cov,_=analysis
        fits_rows+=fr
        for row in rr:
            k=int(np.argmin(abs(CENTERS-row['lag_center_days'])))
            anchor=int(np.argmin(abs(CENTERS-2371)))
            use=contributing[:,k]|contributing[:,anchor]
            td=theta[use,k]-theta[use,anchor]
            row['rise_plate_jack_sigma']=np.sqrt((len(td)-1)/len(td)*np.sum((td-td.mean())**2))
        rises+=rr
        bootstrap_outputs[f'cov_{band}']=cov
        bootstrap_outputs[f'replicates_{band}']=reps
        bootstrap_outputs[f'objids_{band}']=oid
        selected[band]=(a,oid,z[keep],lum[keep],adopted[band],kernel)
        for key,value in zip(['nodes','weights','counts'],kernel):
            bootstrap_outputs[f'lag_{key}_{band}']=value
        for method in ['object_mean','pooled','object_median','positive_median']:
            v=esf.ensemble_second_moment(a,method=method)
            for k in range(endpoint+1):
                comparisons.append(dict(band=band,sample='fixed',method=method,lag_center_days=CENTERS[k],
                                        sf2_mag2=v[k],sf_mag=np.sqrt(max(v[k],0)),n_objects=len(a)))
        conv=np.percentile(reps[:2500],[16,84],axis=0)
        full=np.percentile(reps,[16,84],axis=0)
        print(f'{band}: {len(a)} fixed objects, through {CENTERS[endpoint]:.0f} d; SF {table.sf_mag.iloc[-1]:.4f} '
              f'[{table.sf_lo_mag.iloc[-1]:.4f},{table.sf_hi_mag.iloc[-1]:.4f}]; '
              f'bootstrap-bound convergence {abs(full-conv).max():.4f} mag',flush=True)
    fixed=pd.concat(rows,ignore_index=True)
    fixed.to_csv(output/'structure_function_fixed_population.csv',index=False)
    pd.DataFrame(comparisons).to_csv(output/'structure_function_aggregation_comparison.csv',index=False)
    np.savez_compressed(output/'ensemble_sf_covariance.npz',**bootstrap_outputs)
    _object_groups(selected,output,fits_rows,rises)
    pd.DataFrame(fits_rows).to_csv(output/'structure_function_object_fits.csv',index=False)
    pd.DataFrame(rises).to_csv(output/'structure_function_object_rises.csv',index=False)
    (output/'adopted_selection.json').write_text(json.dumps(adopted,indent=2)+'\n')
    for path in output.glob('*.csv'):
        shutil.copy2(path,_ROOT/'data'/path.name)
    shutil.copy2(output/'ensemble_sf_covariance.npz',_ROOT/'data/ensemble_sf_covariance.npz')
    print(f'Adopted object-first outputs: {output}',flush=True)


def _object_groups(selected, output, fitrows, rises):
    ids=np.unique(np.concatenate([v[1] for v in selected.values()]))
    properties=esf.load_properties(ids).reindex(ids)
    valid=properties.notna().all(axis=1)
    edges={n:np.quantile(properties.loc[valid,n],[0,.5,1]) for n in ['LOGMBH','LOGLEDD_RATIO']}
    records,grid_records,matching,inputs=[],[],[],{}
    for bi,band in enumerate(BANDS):
        acc,oid,z,lum,_,kernel=selected[band]
        p=properties.reindex(oid)
        good=p.notna().all(axis=1).to_numpy()
        cell,ncell=_cell_index(z,lum,2)
        masks={f'{ml}_{el}':good & mm & em
               for ml,mm in [('lowmbh',p.LOGMBH.to_numpy()<edges['LOGMBH'][1]),
                             ('highmbh',p.LOGMBH.to_numpy()>=edges['LOGMBH'][1])]
               for el,em in [('lowedd',p.LOGLEDD_RATIO.to_numpy()<edges['LOGLEDD_RATIO'][1]),
                             ('highedd',p.LOGLEDD_RATIO.to_numpy()>=edges['LOGLEDD_RATIO'][1])]}
        counts=np.stack([np.bincount(cell[m],minlength=ncell) for m in masks.values()])
        target=(counts.min(axis=0)>=5).astype(float)
        for group,nn in zip(masks,counts):
            for c,number in enumerate(nn):
                matching.append(dict(band=band,sample=group,cell=c,n_objects=int(number),
                                     common_cell=bool(target[c]),minimum_objects=5))
        print(f'{band}: homogeneous groups {[int(m.sum()) for m in masks.values()]}; '
              f'{int(target.sum())}/{ncell} z-L cells with >=5/group',flush=True)
        for gi,(name,keep) in enumerate(masks.items()):
            if keep.sum()<20:
                continue
            a=acc[keep]
            tab,reps=_object_summary(a,z[keep],lum[keep],n_boot=3000,seed=3300+10*bi+gi)
            tab['sample'],tab['band']=name,band
            tab['mbh_split'],tab['edd_split']=edges['LOGMBH'][1],edges['LOGLEDD_RATIO'][1]
            mi=0 if name.startswith('lowmbh') else 1
            ei=0 if name.endswith('lowedd') else 1
            tab['mbh_lo'],tab['mbh_hi']=edges['LOGMBH'][mi:mi+2]
            tab['edd_lo'],tab['edd_hi']=edges['LOGLEDD_RATIO'][ei:ei+2]
            tab=tab.rename(columns={'sf_mag':'sf_raw_mag','sf_lo_mag':'sf_raw_lo_mag',
                                    'sf_hi_mag':'sf_raw_hi_mag','sf_err_mag':'sf_raw_err_mag'})
            if target.any():
                occupied=np.bincount(cell[keep],minlength=ncell)
                weights=np.divide(target,occupied,out=np.zeros_like(target),where=occupied>0)[cell[keep],None]
                matched,mreps=_object_summary(a,z[keep],lum[keep],n_boot=3000,seed=3300+10*bi+gi,weights=weights)
                for old,new in [('sf_mag','sf_matched_mag'),('sf_lo_mag','sf_matched_lo_mag'),
                                ('sf_hi_mag','sf_matched_hi_mag'),('sf_err_mag','sf_matched_err_mag')]:
                    tab[new]=matched[old]
                tab['n_objects_matched']=int((weights[:,0]>0).sum())
                tab['cell_coverage']=1.
                tab['z_med_matched'],tab['logl_med_matched']=matched.z_med,matched.logl_med
            sub=tuple(v[keep] for v in kernel)
            analysis=_object_fit(a,reps,name,band,sub)
            if analysis:
                fitrows+=analysis[0]
                rises+=analysis[1]
                inputs[f'p_{band}_{name}'],inputs[f'pcov_{band}_{name}']=analysis[2],analysis[4]
            records.append(tab)
            inputs[f'keep_{band}_{name}']=keep
        me=np.quantile(properties.loc[valid,'LOGMBH'],np.linspace(0,1,4))
        ee=np.quantile(properties.loc[valid,'LOGLEDD_RATIO'],np.linspace(0,1,4))
        for i in range(3):
            for j in range(3):
                keep=good&(p.LOGMBH.to_numpy()>=me[i])&((p.LOGMBH.to_numpy()<me[i+1]) if i<2 else (p.LOGMBH.to_numpy()<=me[i+1]))
                keep&=(p.LOGLEDD_RATIO.to_numpy()>=ee[j])&((p.LOGLEDD_RATIO.to_numpy()<ee[j+1]) if j<2 else (p.LOGLEDD_RATIO.to_numpy()<=ee[j+1]))
                if keep.sum()<20:
                    continue
                a=acc[keep]
                tab,reps=_object_summary(a,z[keep],lum[keep],n_boot=2000,seed=4400+100*bi+3*i+j)
                tab['grid'],tab['i_mbh'],tab['j_edd'],tab['band']='q3x3',i,j,band
                tab['mbh_lo'],tab['mbh_hi'],tab['edd_lo'],tab['edd_hi']=me[i],me[i+1],ee[j],ee[j+1]
                tab=tab.rename(columns={'sf_mag':'sf_raw_mag','sf_lo_mag':'sf_raw_lo_mag',
                                        'sf_hi_mag':'sf_raw_hi_mag','sf_err_mag':'sf_raw_err_mag'})
                analysis=_object_fit(a,reps,f'grid_{i}_{j}',band,tuple(v[keep] for v in kernel))
                if analysis:
                    fitrows+=analysis[0]
                    rises+=analysis[1]
                grid_records.append(tab)
    pd.DataFrame(matching).to_csv(output/'structure_function_group_matching.csv',index=False)
    pd.concat(records,ignore_index=True).to_csv(output/'structure_function_matched_mbh_edd.csv',index=False)
    pd.concat(grid_records,ignore_index=True).to_csv(output/'structure_function_mbh_edd_grid.csv',index=False)
    return inputs


def object_first_refit():
    import shutil
    from plate_completeness import ADOPTED_SELECTION, load_completeness_lightcurves
    output=esf.CACHE/'object_first'
    with np.load(output/'object_accumulator.npz') as f:
        acc,ids=f['acc'],f['ids'].astype(str)
    sample=pd.read_parquet(_ROOT/'data/plate_completeness_sample.parquet')
    sample['OBJID']=sample.OBJID.astype(str)
    cats=sample[sample.band.eq('g')].set_index('OBJID').reindex(ids)
    z,lum=cats.z.to_numpy(float),cats.loglbol.to_numpy(float)
    zmap=pd.Series(z,index=ids)
    with np.load(output/'ensemble_sf_covariance.npz') as f:
        boot={k:f[k] for k in f.files}
    oldrises=pd.read_csv(output/'structure_function_object_rises.csv')
    selected,available,figure,fitrows,rises,checks={},{},{},[],[],[]
    for bi,band in enumerate(BANDS):
        oid=boot[f'objids_{band}'].astype(str)
        indices=pd.Index(ids).get_indexer(oid)
        assert (indices>=0).all()
        reps=boot[f'replicates_{band}']
        a=acc[indices,bi,:reps.shape[1]]
        if f'lag_nodes_{band}' not in boot:
            nt=esf.clean_nightly(load_completeness_lightcurves(oid).query('band == @band'))[0]
            kernel=_object_lag_kernel(nt,oid,zmap,a.shape[1])
            np.testing.assert_array_equal(kernel[2],a[...,0])
            for key,value in zip(['nodes','weights','counts'],kernel):
                boot[f'lag_{key}_{band}']=value
            np.savez_compressed(output/'ensemble_sf_covariance.npz',**boot)
        kernel=tuple(boot[f'lag_{key}_{band}'] for key in ['nodes','weights','counts'])
        fr,rr,p,_,pcov=_object_fit(a,reps,'fixed',band,kernel)
        fitrows+=fr
        for row in rr:
            previous=oldrises[oldrises['sample'].eq('fixed')&oldrises.band.eq(band)&np.isclose(oldrises.lag_center_days,row['lag_center_days'])].iloc[0]
            row['rise_plate_jack_sigma']=previous.rise_plate_jack_sigma
        rises+=rr
        available[band]=sample[sample.band.eq(band)].set_index('OBJID').reindex(ids)[ADOPTED_SELECTION[band]].fillna(False).to_numpy(bool)
        selected[band]=(a,oid,z[indices],lum[indices],ADOPTED_SELECTION[band],kernel)
        figure[f'p_{band}_fixed'],figure[f'pcov_{band}_fixed']=p,pcov
        nd,wt,counts=kernel
        coarse_w=wt.reshape(*wt.shape[:-1],16,2).sum(axis=-1)
        coarse_n=np.divide((nd*wt).reshape(*wt.shape[:-1],16,2).sum(axis=-1),coarse_w,
                           out=np.zeros_like(coarse_w),where=coarse_w>0)
        coarse=(coarse_n,coarse_w,counts)
        fine_drw,fine_pl,_=_object_sampled_models(kernel)
        coarse_drw,coarse_pl,_=_object_sampled_models(coarse)
        for row in fr:
            if row['model'] not in ['drw_short','drw_all','powerlaw_all']:
                continue
            par=[row['amplitude'],row['timescale_or_slope']]
            fm,cm,center=(fine_pl,coarse_pl,powerlaw) if row['model']=='powerlaw_all' else (fine_drw,coarse_drw,drw)
            x=CENTERS[:a.shape[1]]
            checks.append(dict(band=band,model=row['model'],max_16_32_difference_mag=float(abs(fm(x,*par)-cm(x,*par)).max()),
                               max_center_sampled_difference_mag=float(abs(fm(x,*par)-center(x,*par)).max()),
                               subdivisions=32))
        print(f'{band}: sampled-lag DRW {p}; kernel verified against all pair counts',flush=True)
    figure|=_object_groups(selected,output,fitrows,rises)
    pd.DataFrame(fitrows).to_csv(output/'structure_function_object_fits.csv',index=False)
    pd.DataFrame(rises).to_csv(output/'structure_function_object_rises.csv',index=False)
    pd.DataFrame(checks).to_csv(output/'structure_function_lag_integration_check.csv',index=False)
    rows=[dict(band=band,lag_center_days=c,n_objects=int(u.sum()),z_med=np.median(z[u]) if u.any() else np.nan)
          for bi,band in enumerate(BANDS) for c,u in zip(CENTERS,(available[band][:,None]&(acc[:,bi,:,0]>0)).T)]
    pd.DataFrame(rows).to_csv(output/'structure_function_varying_membership_redshift.csv',index=False)
    np.savez_compressed(esf.cache_path('object_first/figure_inputs.npz'),**figure)
    for name in ['structure_function_object_fits.csv','structure_function_object_rises.csv',
                 'structure_function_lag_integration_check.csv','structure_function_matched_mbh_edd.csv',
                 'structure_function_mbh_edd_grid.csv','structure_function_group_matching.csv',
                 'structure_function_varying_membership_redshift.csv','ensemble_sf_covariance.npz']:
        shutil.copy2(output/name,_ROOT/'data'/name)
    print('Sampled-lag refit complete; empirical SF and selected objects unchanged',flush=True)


if __name__ == "__main__":
    esf.check_flags(__file__)
    import sys
    if "--object-first" in sys.argv or len(sys.argv) == 1:
        object_first_results()
    elif "--refit-object-first" in sys.argv:
        object_first_refit()
    elif "--literature-choices" in sys.argv:
        literature_choices()
