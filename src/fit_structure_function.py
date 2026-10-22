from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

import ensemble_structure_function as esf
from ensemble_structure_function import BANDS, CENTERS, EDGES, NB, accumulate

_ROOT = Path(__file__).resolve().parent.parent
SHORT_MAX_D = 3000.0
PLATEAU_MIN_D = 400.0  # plateau fit start (CCD lags 422-2371 d)
LONG_MIN_D = 11 * 365.25  # 11 rest-frame yr; excess bins beyond
STONE_FITS = "data/stone2022_ensemble_sf_psd.fits.gz"
STONE_LC = "data/stone2022_lightcurves.fits.gz"
STONE_FLOOR_END_D = {"g": 5.0, "r": 20.0, "i": 40.0}
FIXED_CSV = "data/structure_function_fixed_population.csv"


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
POPDRW_LAM_OBS = {"g": 4800.0, "r": 6250.0}
GROUPS = ("lowmbh_lowedd", "lowmbh_highedd", "highmbh_lowedd", "highmbh_highedd")


def bootstrap_plan(band, name):
    """(n_boot, seed) of the fiducial sample, the four groups or a grid_i_j cell"""
    bi = BANDS.index(band)
    match name.split("_"):
        case ["fiducial"]:
            return 5000, 1927 + bi
        case ["grid", i, j]:
            return 2000, 4400 + 100 * bi + 3 * int(i) + int(j)
        case _:
            return 3000, 3300 + 10 * bi + GROUPS.index(name)


def macleod_drw(coef, band, z, mi, logmbh):
    x = np.column_stack([np.ones(len(z)), np.log10(POPDRW_LAM_OBS[band]/(1+np.asarray(z))/4000),
                         np.asarray(mi)+23, np.asarray(logmbh)-9])
    return [x @ np.asarray(coef[c]) for c in ("sfinf", "tau")]


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
    """Rest-frame pair lags of each quasar within each bin, on log sub-bins

    Returns
        nodes, mean lag per sub-bin (quasar, bin, subdivisions), days
        weights, fraction of the bin's pairs per sub-bin
        count, pairs per quasar and bin
    """
    nodes = np.zeros((len(ids), n_bins, subdivisions))
    weights = np.zeros_like(nodes)
    count = np.zeros((len(ids), n_bins), dtype=int)
    edges = np.geomspace(EDGES[0], EDGES[n_bins], n_bins*subdivisions+1)
    groups = dict(tuple(nightly.groupby('OBJID', sort=False)))
    for q, oid in enumerate(ids):
        t = groups[oid].time.to_numpy(float)
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


def _object_rises(a, reps, name, band):
    """SF rise of each bin beyond 11 rest-frame years over the bin nearest 2371 d

    reps: bootstrap SF replicates (n_boot, bin), mag
    """
    sf = _object_amplitude(esf.ensemble_second_moment(a))
    k0 = int(np.argmin(abs(CENTERS[:len(sf)]-2371)))
    rise = []
    for k in np.flatnonzero(CENTERS[:len(sf)] > LONG_MIN_D):
        diff = reps[:, k]-reps[:, k0]
        lo, hi = np.percentile(diff, [16, 84])
        rise.append(dict(sample=name, band=band, lag_center_days=CENTERS[k],
                         sf_anchor_mag=sf[k0], sf_mag=sf[k], rise_mag=sf[k]-sf[k0],
                         rise_lo_mag=lo, rise_hi_mag=hi,
                         rise_bootstrap_sigma=float(diff.std(ddof=1)),
                         ratio=sf[k]/sf[k0]))
    return rise


def _object_completeness_sweep(acc, ids, sample, z, lum, outdir, n_boot=10000):
    from plate_completeness import ADOPTED_SELECTION
    rows, compare, panels, matched_rows, shape_rows, criteria = [], [], {}, [], [], []
    candidates = [f'{kind}_cut_{margin}' for kind in ['plate','all'] for margin in ['025','050','075','100','125']]+['all_nocut','all_lim050','all_cut_000','all_lim095']
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
        pairs = [(i,j) for i in range(10) for j in range(i+1,10) if candidates[i].split('_')[0]==candidates[j].split('_')[0]]
        for ci,cj in pairs+[(i,5) for i in range(10,len(candidates))]:
            delta = reps[:,ci]-reps[:,cj]
            for k in long:
                q = np.percentile(delta[:,k],[16,84])
                compare.append(dict(band=band,selection=candidates[ci],comparison=candidates[cj],lag_center_days=CENTERS[k],
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
            mw=w[:,5:10]*wc[:,:,None]
            mpoint=_object_amplitude(esf.ensemble_second_moment(expanded[:,5:10],weights=mw))
            mbounds,mreps=esf.object_bootstrap(expanded[:,5:10],n_boot=n_boot,seed=927+bi,weights=mw)
            mbounds=_object_amplitude(mbounds)
            for c,name in enumerate(candidates[5:10]):
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
        for ci,others in [*((c,range(c+1,10)) for c in range(5,8)),(10,[5]),(11,[5]),(12,range(5,10)),(13,[5])]:
            all_stable=True
            for cj in others:
                delta=reps[:,ci,long]-reps[:,cj,long]
                se=delta.std(axis=0,ddof=1)
                centered=(delta-delta.mean(axis=0))/se
                threshold=np.percentile(np.max(abs(centered),axis=1),95)
                statistic=float(np.max(abs(pointsf[ci,long]-pointsf[cj,long])/se))
                skipped = masks[:,cj].sum()<50
                all_stable &= statistic <= threshold or skipped
                criteria.append(dict(band=band,selection=candidates[ci],comparison=candidates[cj],
                                     maximum_standardized_difference=statistic,threshold95=threshold,
                                     simultaneous_p=float((np.max(abs(centered),axis=1)>=statistic).mean()),
                                     n_comparison=int(masks[:,cj].sum()),skipped=bool(skipped),bootstrap_draws=n_boot))
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
    import json
    import shutil
    from plate_completeness import accumulator_signature, load_completeness_lightcurves

    output=esf.CACHE/'object_first'
    output.mkdir(parents=True,exist_ok=True)
    sample=pd.read_parquet(_ROOT/'data/plate_completeness_sample.parquet')
    sample['OBJID']=sample.OBJID.astype(str)
    ids=np.sort(sample.loc[sample.all_nocut,'OBJID'].unique()).astype(str)
    cats=sample[sample.band.eq('g')].set_index('OBJID').reindex(ids)
    z,lum=cats.z.to_numpy(float),cats.loglbol.to_numpy(float)
    zmap=pd.Series(z,index=ids)
    signature=accumulator_signature(ids)
    cache=output/'object_accumulator.npz'
    completed=0
    acc=np.zeros((len(ids),len(BANDS),NB,6))
    if cache.exists():
        with np.load(cache) as f:
            if str(f['signature'])==signature:
                acc=f['acc']
                completed=int(f['completed']) if 'completed' in f else len(ids)
    for start in range(completed,len(ids),1000):
        stop=min(start+1000,len(ids))
        part=ids[start:stop]
        acc[start:stop]=accumulate(esf.clean_nightly(load_completeness_lightcurves(part)),part,zmap,resid=True)
        np.savez_compressed(cache,acc=acc,ids=ids,signature=signature,completed=stop)
        print(f'Accumulated {stop}/{len(ids)} objects',flush=True)
    adopted=_object_completeness_sweep(acc,ids,sample,z,lum,output)
    for band in BANDS:
        sample.loc[sample.band.eq(band),'adopted_keep']=sample.loc[sample.band.eq(band),adopted[band]]
    sample.to_parquet(_ROOT/'data/plate_completeness_sample.parquet',index=False)
    rows,comparisons,rises,redshift=[],[],[],[]
    bootstrap_outputs={}
    selected={}
    for bi,band in enumerate(BANDS):
        s=sample[sample.band.eq(band)].set_index('OBJID').reindex(ids)
        available=s[adopted[band]].fillna(False).to_numpy(bool)
        endpoint=12 if band=='g' else 13
        keep=available&(acc[:,bi,:endpoint+1,0]>0).all(axis=1)
        a=acc[keep,bi,:endpoint+1]
        oid=ids[keep]
        table,reps=_object_summary(a,z[keep],lum[keep],*bootstrap_plan(band,"fiducial"))
        table['sample']='fixed'
        table['band']=band
        table['selection']=adopted[band]
        nt=esf.clean_nightly(load_completeness_lightcurves(oid).query('band == @band'))
        nights=nt.groupby('OBJID').time
        span,edge=(nights.max()-nights.min()).max(),EDGES[endpoint]
        redshift.append(dict(band=band,n=len(oid),z_min=z[keep].min(),z_median=np.median(z[keep]),z_max=z[keep].max(),
                             final_bin_lower_edge_days=edge,max_observed_separation_days=span,implied_z_max=span/edge-1))
        rows.append(table)
        kernel=_object_lag_kernel(nt,oid,zmap,a.shape[1])
        np.testing.assert_array_equal(kernel[2],a[...,0])
        rises+=_object_rises(a,reps,'fixed',band)
        bootstrap_outputs[f'replicates_{band}']=reps
        bootstrap_outputs[f'objids_{band}']=oid
        selected[band]=(a,oid,z[keep],lum[keep],adopted[band])
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
    _object_groups(selected,output,rises)
    pd.DataFrame(rises).to_csv(output/'structure_function_object_rises.csv',index=False)
    pd.DataFrame(redshift).to_csv(output/'structure_function_fiducial_redshift.csv',index=False)
    (output/'adopted_selection.json').write_text(json.dumps(adopted,indent=2)+'\n')
    for name in ['fixed_population','aggregation_comparison','object_rises','group_matching','matched_mbh_edd',
                 'mbh_edd_grid','completeness_sweep','completeness_stability','completeness_shapes','completeness_matched',
                 'completeness_differences','fiducial_redshift']:
        shutil.copy2(output/f'structure_function_{name}.csv',_ROOT/'data'/f'structure_function_{name}.csv')
    shutil.copy2(output/'ensemble_sf_covariance.npz',_ROOT/'data/ensemble_sf_covariance.npz')
    print(f'Adopted object-first outputs: {output}',flush=True)


def _object_groups(selected, output, rises):
    ids=np.unique(np.concatenate([v[1] for v in selected.values()]))
    properties=esf.load_properties(ids).reindex(ids)
    valid=properties.notna().all(axis=1)
    edges={n:np.quantile(properties.loc[valid,n],[0,.5,1]) for n in ['LOGMBH','LOGLEDD_RATIO']}
    records,grid_records,matching=[],[],[]
    for band in BANDS:
        acc,oid,z,lum,_=selected[band]
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
        for name,keep in masks.items():
            if keep.sum()<20:
                continue
            a=acc[keep]
            tab,reps=_object_summary(a,z[keep],lum[keep],*bootstrap_plan(band,name))
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
                matched,mreps=_object_summary(a,z[keep],lum[keep],*bootstrap_plan(band,name),weights=weights)
                for old,new in [('sf_mag','sf_matched_mag'),('sf_lo_mag','sf_matched_lo_mag'),
                                ('sf_hi_mag','sf_matched_hi_mag'),('sf_err_mag','sf_matched_err_mag')]:
                    tab[new]=matched[old]
                tab['n_objects_matched']=int((weights[:,0]>0).sum())
                tab['cell_coverage']=1.
                tab['z_med_matched'],tab['logl_med_matched']=matched.z_med,matched.logl_med
            rises+=_object_rises(a,reps,name,band)
            records.append(tab)
        me=np.quantile(properties.loc[valid,'LOGMBH'],np.linspace(0,1,4))
        ee=np.quantile(properties.loc[valid,'LOGLEDD_RATIO'],np.linspace(0,1,4))
        for i in range(3):
            for j in range(3):
                keep=good&(p.LOGMBH.to_numpy()>=me[i])&((p.LOGMBH.to_numpy()<me[i+1]) if i<2 else (p.LOGMBH.to_numpy()<=me[i+1]))
                keep&=(p.LOGLEDD_RATIO.to_numpy()>=ee[j])&((p.LOGLEDD_RATIO.to_numpy()<ee[j+1]) if j<2 else (p.LOGLEDD_RATIO.to_numpy()<=ee[j+1]))
                if keep.sum()<20:
                    continue
                a=acc[keep]
                tab,reps=_object_summary(a,z[keep],lum[keep],*bootstrap_plan(band,f'grid_{i}_{j}'))
                tab['grid'],tab['i_mbh'],tab['j_edd'],tab['band']='q3x3',i,j,band
                tab['mbh_lo'],tab['mbh_hi'],tab['edd_lo'],tab['edd_hi']=me[i],me[i+1],ee[j],ee[j+1]
                tab=tab.rename(columns={'sf_mag':'sf_raw_mag','sf_lo_mag':'sf_raw_lo_mag',
                                        'sf_hi_mag':'sf_raw_hi_mag','sf_err_mag':'sf_raw_err_mag'})
                rises+=_object_rises(a,reps,f'grid_{i}_{j}',band)
                grid_records.append(tab)
    pd.DataFrame(matching).to_csv(output/'structure_function_group_matching.csv',index=False)
    pd.concat(records,ignore_index=True).to_csv(output/'structure_function_matched_mbh_edd.csv',index=False)
    pd.concat(grid_records,ignore_index=True).to_csv(output/'structure_function_mbh_edd_grid.csv',index=False)


def object_first_refit():
    import shutil
    from plate_completeness import ADOPTED_SELECTION
    output=esf.CACHE/'object_first'
    with np.load(output/'object_accumulator.npz') as f:
        acc,ids=f['acc'],f['ids'].astype(str)
    sample=pd.read_parquet(_ROOT/'data/plate_completeness_sample.parquet')
    sample['OBJID']=sample.OBJID.astype(str)
    cats=sample[sample.band.eq('g')].set_index('OBJID').reindex(ids)
    z,lum=cats.z.to_numpy(float),cats.loglbol.to_numpy(float)
    with np.load(output/'ensemble_sf_covariance.npz') as f:
        boot={k:f[k] for k in f.files}
    selected,available,rises={},{},[]
    for bi,band in enumerate(BANDS):
        oid=boot[f'objids_{band}'].astype(str)
        indices=pd.Index(ids).get_indexer(oid)
        assert (indices>=0).all()
        reps=boot[f'replicates_{band}']
        a=acc[indices,bi,:reps.shape[1]]
        rises+=_object_rises(a,reps,'fixed',band)
        available[band]=sample[sample.band.eq(band)].set_index('OBJID').reindex(ids)[ADOPTED_SELECTION[band]].fillna(False).to_numpy(bool)
        selected[band]=(a,oid,z[indices],lum[indices],ADOPTED_SELECTION[band])
    _object_groups(selected,output,rises)
    pd.DataFrame(rises).to_csv(output/'structure_function_object_rises.csv',index=False)
    rows=[dict(band=band,lag_center_days=c,n_objects=int(u.sum()),z_med=np.median(z[u]) if u.any() else np.nan)
          for bi,band in enumerate(BANDS) for c,u in zip(CENTERS,(available[band][:,None]&(acc[:,bi,:,0]>0)).T)]
    pd.DataFrame(rows).to_csv(output/'structure_function_varying_membership_redshift.csv',index=False)
    for name in ['structure_function_object_rises.csv','structure_function_matched_mbh_edd.csv',
                 'structure_function_mbh_edd_grid.csv','structure_function_group_matching.csv',
                 'structure_function_varying_membership_redshift.csv','ensemble_sf_covariance.npz']:
        shutil.copy2(output/name,_ROOT/'data'/name)


if __name__ == "__main__":
    esf.check_flags(__file__)
    import sys
    if "--object-first" in sys.argv:
        object_first_results()
    elif "--refit-object-first" in sys.argv:
        object_first_refit()
    elif "--literature-choices" in sys.argv:
        literature_choices()
    else:
        raise SystemExit("usage: --object-first | --refit-object-first | --literature-choices")
