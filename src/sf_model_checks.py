import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

import ensemble_structure_function as esf
import fit_structure_function as fsf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data'
LOGX = np.linspace(-6, 6, 481)
SLOPE_ALPHA = np.round(np.arange(-0.5, 2.901, 0.05), 8)
ALPHA = SLOPE_ALPHA[SLOPE_ALPHA < 2]


def psd_sf2(x, alpha, beta):
    # unnormalized SF^2 of A/(u^alpha+u^beta), v = x*u; v^(2-m) weight holds the low-frequency singularity
    s, m, k = x**(beta-alpha), min(alpha, beta), max(x, 1.)
    d = lambda v: 1/(s*v**(alpha-m)+v**(beta-m))
    w = lambda v: d(v)/v**m
    q = lambda f, lo, hi, **kw: quad(f, lo, hi, epsabs=0, epsrel=1e-9, limit=1000, **kw)[0]
    low = q(lambda v: .5*np.sinc(v/2/np.pi)**2*d(v), 0, min(x, 1.), weight='alg', wvar=(2-m, 0))
    low += q(lambda v: 2*np.sin(v/2)**2*w(v), x, 1) if x < 1 else q(w, 1, k)-q(w, 1, k, weight='cos', wvar=1.)
    tail = q(w, k, np.inf)
    return x**(beta-1)*(low+tail-quad(w, k, np.inf, weight='cos', wvar=1., epsabs=1e-10*tail, limlst=200)[0])


def cached_grid(path, alphas, logx, f):
    if path.exists():
        with np.load(path) as c:
            if np.array_equal(c['alpha'], alphas) and np.array_equal(c['logx'], logx):
                return RegularGridInterpolator((alphas, logx), c['values'], bounds_error=True)
    values = np.array([[f(10**x, a) for x in logx] for a in alphas])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Nonpositive or nonfinite PSD transform')
    np.savez_compressed(path, alpha=alphas, logx=logx, values=values)
    return RegularGridInterpolator((alphas, logx), values, bounds_error=True)


def psd_grid(beta=None):
    if beta is None:
        g = psd_grid(2.)
        return RegularGridInterpolator((ALPHA, g.grid[1]), g.values[:len(ALPHA)], bounds_error=True)
    return cached_grid(esf.cache_path(f'model_checks/psd_integrals_beta{beta:g}.npz'), SLOPE_ALPHA, LOGX[np.abs(LOGX) < 4.01],
                       lambda x, a: psd_sf2(x, a, beta))


def samples():
    with np.load(OUT/'ensemble_sf_covariance.npz') as f:
        boot = {k:f[k] for k in f.files}
    with np.load(OUT/'structure_function_sampling_covariance.npz') as f:
        object_data = {k:f[k] for k in f.files if k.startswith('fiducial_ids_') or k.startswith('fiducial_acc_')}
    all_ids = np.unique(np.concatenate([boot[f'objids_{b}'].astype(str) for b in esf.BANDS]))
    prop = esf.load_properties(all_ids).reindex(all_ids)
    med = {c:np.quantile(prop[c].dropna(), [0,.5,1]) for c in ['LOGMBH','LOGLEDD_RATIO']}
    thirds = {c:np.quantile(prop[c].dropna(), np.linspace(0,1,4)) for c in med}
    catalog = pd.read_parquet(OUT/'plate_completeness_sample.parquet')
    catalog['OBJID'] = catalog.OBJID.astype(str)
    for band in esf.BANDS:
        oid = boot[f'objids_{band}'].astype(str)
        pos = pd.Index(object_data[f'fiducial_ids_{band}'].astype(str)).get_indexer(oid)
        if (pos < 0).any():
            raise ValueError('Missing fiducial accumulator objects')
        nb = boot[f'replicates_{band}'].shape[1]
        a = object_data[f'fiducial_acc_{band}'][pos,:nb]
        p = prop.reindex(oid).copy()
        p['z'] = catalog[catalog.band.eq(band)].set_index('OBJID').reindex(oid).z
        kernel = tuple(boot[f'lag_{key}_{band}'] for key in ['nodes','weights','counts'])
        masks = {'fiducial':np.ones(len(oid),bool)}
        for i,ml in enumerate(['lowmbh','highmbh']):
            for j,el in enumerate(['lowedd','highedd']):
                masks[f'{ml}_{el}'] = ((p.LOGMBH >= med['LOGMBH'][i]) &
                    (p.LOGMBH <= med['LOGMBH'][i+1] if i else p.LOGMBH < med['LOGMBH'][i+1]) &
                    (p.LOGLEDD_RATIO >= med['LOGLEDD_RATIO'][j]) &
                    (p.LOGLEDD_RATIO <= med['LOGLEDD_RATIO'][j+1] if j else p.LOGLEDD_RATIO < med['LOGLEDD_RATIO'][j+1])).to_numpy()
        for i in range(3):
            for j in range(3):
                m = ((p.LOGMBH >= thirds['LOGMBH'][i]) &
                     (p.LOGMBH <= thirds['LOGMBH'][i+1] if i==2 else p.LOGMBH < thirds['LOGMBH'][i+1]) &
                     (p.LOGLEDD_RATIO >= thirds['LOGLEDD_RATIO'][j]) &
                     (p.LOGLEDD_RATIO <= thirds['LOGLEDD_RATIO'][j+1] if j==2 else p.LOGLEDD_RATIO < thirds['LOGLEDD_RATIO'][j+1])).to_numpy()
                if m.sum() >= 20:
                    masks[f'grid_{i}_{j}'] = m
        for name,keep in masks.items():
            yield band,name,a[keep],p.iloc[np.flatnonzero(keep)],tuple(v[keep] for v in kernel)


def covariance(reps, rho=None):
    sd = np.std(reps,axis=0,ddof=1)
    corr,_,s = esf.ledoit_wolf(reps/sd)
    if rho is not None:
        corr = (1-rho)*s+rho*np.trace(s)/len(s)*np.eye(len(s))
    return corr*np.outer(sd,sd)


def fit_psd(a, kernel, grid, seed, start=fsf.PLATEAU_MIN_D, beta=2., rho=None):
    alphas,(lo,hi) = grid.grid[0],grid.grid[1][[0,-1]]
    values = esf.object_second_moments(a)
    sf = np.sqrt(np.maximum(values.mean(axis=0),0))
    _,reps2 = esf.object_bootstrap(a,n_boot=5000,seed=seed)
    reps = np.sqrt(np.maximum(reps2,0))
    use = (fsf.CENTERS[:len(sf)] >= start) & (sf > 0)
    cov = covariance(reps[:,use],rho)
    factor = cho_factor(cov)
    y = sf[use]
    cy = cho_solve(factor,y)
    nodes,weights = kernel[:2]
    weight = weights.mean(axis=0)[use]
    node = np.divide((nodes*weights).mean(axis=0)[use],weight,
                     out=np.ones_like(weight),where=weight>0)
    lognode = np.log10(node)
    def trial(logtau,alpha):
        if alpha == 0 and beta == 2:
            shape = -np.expm1(-node/10**logtau)
        else:
            x = np.clip(lognode-logtau,lo,hi)
            shape = grid(np.stack([np.full(x.size,alpha),x.ravel()],axis=1)).reshape(x.shape)
        u = np.sqrt((weight*shape).sum(axis=1))
        cu = cho_solve(factor,u)
        amp = (u@cy)/(u@cu)
        r = y-amp*u
        return float(r@cho_solve(factor,r)),amp,amp*u
    def profile(alpha):
        ts = np.linspace(1,5,65)
        costs = np.array([trial(t,alpha)[0] for t in ts])
        k = costs.argmin()
        best = minimize_scalar(lambda t:trial(t,alpha)[0],
                               bounds=(ts[max(k-1,0)],ts[min(k+1,len(ts)-1)]),method='bounded')
        candidates = [best.x,ts[0],ts[-1]]
        t = min(candidates,key=lambda q:trial(q,alpha)[0])
        cost,amp,pred = trial(t,alpha)
        return dict(alpha=float(alpha),chi2=cost,tau_days=10**t,sf_inf_mag=amp),pred
    profiles = [profile(alpha)[0] for alpha in alphas]
    best = min(profiles,key=lambda d:d['chi2'])
    drw,pred_drw = profile(0.)
    _,pred_bend = profile(best['alpha'])
    rows=[]
    for model,result,npar in [('drw',drw,2),('bending_psd',best,3)]:
        ndof = int(use.sum())-npar
        accepted = [p['alpha'] for p in profiles if p['chi2'] <= best['chi2']+1] if npar==3 else []
        rows.append(dict(model=model,**result,n_bins=int(use.sum()),dof=ndof,
                         reduced_chi2=result['chi2']/ndof,pvalue=chi2.sf(result['chi2'],ndof),
                         delta_chi2_from_drw=drw['chi2']-result['chi2'],
                         alpha_profile_lo=min(accepted) if accepted else np.nan,
                         alpha_profile_hi=max(accepted) if accepted else np.nan,
                         alpha_boundary=model=='bending_psd' and best['alpha'] in [alphas[0],alphas[-1]],
                         alpha_profile_truncated=bool(accepted) and (min(accepted)==alphas[0] or max(accepted)==alphas[-1]),
                         tau_boundary=result['tau_days'] < 10.01 or result['tau_days'] > 99990))
    curves = pd.DataFrame(dict(lag_center_days=fsf.CENTERS[:len(sf)][use],sf_mag=y,
                               sf_err_mag=np.sqrt(np.diag(cov)),drw_mag=pred_drw,bending_mag=pred_bend))
    return rows,profiles,curves


def nested_checks(a, prop, seed):
    values = esf.object_second_moments(a)
    n,nb = values.shape
    masks={'original':np.ones(n,bool)}
    for fraction in [.8,.6]:
        keep=np.ones(n,bool)
        for col in ['LOGMBH','LOGLEDD_RATIO']:
            lo,hi=prop[col].quantile([(1-fraction)/2,(1+fraction)/2])
            keep &= prop[col].between(lo,hi).to_numpy()
        masks[f'central_{int(100*fraction)}_mass_edd']=keep
    lo,hi=prop.z.quantile([.25,.75])
    masks['central_50_redshift']=prop.z.between(lo,hi).to_numpy()
    masks['redshift_060_090']=prop.z.between(.6,.9).to_numpy()
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(n,np.full(n,1/n),size=5000)
    anchor=int(np.argmin(abs(fsf.CENTERS[:nb]-2371)))
    base=np.sqrt(np.maximum(values.mean(axis=0),0))
    bootbase=np.sqrt(np.maximum(counts@values/n,0))
    out,properties,curves=[],[],[]
    for label,mask in masks.items():
        m=int(mask.sum())
        properties.append(dict(selection=label,n_objects=m,
            **{f'{col}_{stat}':getattr(prop.loc[mask,col],stat)() if m else np.nan
               for col in ['LOGMBH','LOGLEDD_RATIO','z'] for stat in ['mean','min','max','std']}))
        if m < 10:
            continue
        sf=np.sqrt(np.maximum(values[mask].mean(axis=0),0))
        denom=counts[:,mask].sum(axis=1)
        good=denom>0
        boot=np.sqrt(np.maximum((counts[good][:,mask]@values[mask])/denom[good,None],0))
        bounds=np.percentile(boot,[16,84],axis=0)
        for k in range(nb):
            curves.append(dict(selection=label,n_objects=m,lag_center_days=fsf.CENTERS[k],
                               sf_mag=sf[k],sf_lo_mag=bounds[0,k],sf_hi_mag=bounds[1,k]))
        for k in np.flatnonzero(fsf.CENTERS[:nb] > fsf.SHORT_MAX_D):
            rise=sf[k]-sf[anchor]
            draws=boot[:,k]-boot[:,anchor]
            delta=draws-(bootbase[good,k]-bootbase[good,anchor])
            q=np.percentile(draws,[2.5,16,84,97.5])
            dq=np.percentile(delta,[2.5,16,84,97.5])
            valid=(boot[:,anchor]>0)&(bootbase[good,anchor]>0)
            ratio=boot[valid,k]/boot[valid,anchor]
            ratio_delta=ratio-bootbase[good][valid,k]/bootbase[good][valid,anchor]
            rq=np.percentile(ratio,[2.5,16,84,97.5])
            rdq=np.percentile(ratio_delta,[2.5,97.5])
            out.append(dict(selection=label,n_objects=m,lag_center_days=fsf.CENTERS[k],
                sf_mag=sf[k],rise_mag=rise,rise_lo95=q[0],rise_lo=q[1],rise_hi=q[2],rise_hi95=q[3],
                difference_from_original_rise=rise-(base[k]-base[anchor]),
                difference_lo95=dq[0],difference_lo=dq[1],difference_hi=dq[2],difference_hi95=dq[3],
                rise_sigma=draws.std(ddof=1),bootstrap_draws=len(draws),
                ratio=sf[k]/sf[anchor],ratio_lo95=rq[0],ratio_lo=rq[1],ratio_hi=rq[2],ratio_hi95=rq[3],
                ratio_difference=sf[k]/sf[anchor]-base[k]/base[anchor],
                ratio_difference_lo95=rdq[0],ratio_difference_hi95=rdq[1],ratio_bootstrap_draws=len(ratio)))
    return out,properties,curves


def influence(a,prop):
    values=esf.object_second_moments(a)
    sf=np.sqrt(np.maximum(values.mean(axis=0),0))
    leave=np.sqrt(np.maximum((values.sum(axis=0)-values)/(len(a)-1),0))
    out=[]
    for k in np.flatnonzero(fsf.CENTERS[:len(sf)] > fsf.SHORT_MAX_D):
        for i,oid in enumerate(prop.index):
            out.append(dict(OBJID=oid,lag_center_days=fsf.CENTERS[k],n_objects=len(a),
                            object_sf2_mag2=values[i,k],sf_mag=sf[k],
                            leave_one_out_sf_mag=leave[i,k],change_mag=leave[i,k]-sf[k],
                            z=prop.z.iloc[i],logmbh=prop.LOGMBH.iloc[i],logedd=prop.LOGLEDD_RATIO.iloc[i]))
    return out


def greedy(a,kernel,stat,steps=2,key=lambda s:s):
    keep,path=np.ones(len(a),bool),[(-1,stat(a,kernel))]
    for _ in range(steps):
        trial={i:stat(a[m],tuple(x[m] for x in kernel))
               for i in np.flatnonzero(keep) for m in [keep&(np.arange(len(a))!=i)]}
        i=min(trial,key=lambda i:key(trial[i]))
        keep[i]=False
        path.append((i,trial[i]))
    return path


def lowmass_influence():
    from plate_completeness import load_completeness_lightcurves
    groups,rows=['lowmbh_lowedd','lowmbh_highedd'],[]
    for band,name,a,prop,kernel in samples():
        if name not in groups:
            continue
        ids=prop.index.to_numpy(str)
        def endpoint(a,kernel):
            reps=np.sqrt(np.maximum(esf.object_bootstrap(a,n_boot=3000,
                                                         seed=3300+10*esf.BANDS.index(band)+groups.index(name))[1],0))
            e=[r for r in fsf._object_fit(a,reps,name,band,kernel)[0] if r['model']=='drw_extrapolation'][-1]
            return e['lag_center_days'],e['excess_mag'],e['excess_err_mag']
        (_,(lag,x,e)),*drops=greedy(a,kernel,endpoint,key=lambda s:s[1]/s[2])
        row=dict(band=band,sample=name,n_objects=len(a),lag_center_days=lag,excess_mag=x,excess_err_mag=e,sigma=x/e)
        for n,(i,(_,dx,de)) in enumerate(drops,1):
            row.update({f'drop{n}_objid':ids[i],f'drop{n}_excess_mag':dx,
                        f'drop{n}_excess_err_mag':de,f'drop{n}_sigma':dx/de})
        nt=esf.clean_nightly(load_completeness_lightcurves(ids).query('band == @band'))[0]
        err,nplates,theta,active=fsf._object_plate_jackknife(nt,ids,prop.z,a,band)
        k=int(np.argmin(abs(fsf.CENTERS-lag)))
        on=active[:,k]
        i=np.argmin(theta[on,k])
        change=theta[on,k][i]-np.sqrt(max(esf.object_second_moments(a)[:,k].mean(),0))
        rows.append(dict(row,plate_jack_sigma_mag=err[k],n_plates=nplates[k],
                         drop_plate=np.sort(nt.loc[nt.sss,'plate'].unique())[on][i],
                         drop_plate_change_mag=change,drop_plate_sigma=(x+change)/e))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'structure_function_lowmass_influence.csv',index=False)
    print(result.to_string(index=False))


def band_ratio():
    with np.load(OUT/'structure_function_sampling_covariance.npz') as f:
        ids=[f[f'fiducial_ids_{b}'] for b in esf.BANDS]
        common=np.intersect1d(*ids)
        g,r=[esf.object_second_moments(f[f'fiducial_acc_{b}'][np.searchsorted(i,common)]) for b,i in zip(esf.BANDS,ids)]
    r=r[:,:g.shape[1]]
    draw=np.random.default_rng(20260928).integers(0,len(common),(5000,len(common)))
    lo,hi=np.percentile(np.sqrt(r[draw].mean(1)/g[draw].mean(1)),[16,84],axis=0)
    sf_g,sf_r=np.sqrt(g.mean(0)),np.sqrt(r.mean(0))
    w=fsf.POPDRW_LAM_OBS['r']/fsf.POPDRW_LAM_OBS['g']
    b_sf,b_tau=fsf.POPDRW_COEF['sfinf'][1],fsf.POPDRW_COEF['tau'][1]
    pd.DataFrame(dict(lag_center_days=fsf.CENTERS[:len(sf_g)],n_objects=len(common),sf_g_mag=sf_g,sf_r_mag=sf_r,
                      ratio=sf_r/sf_g,ratio_lo=lo,ratio_hi=hi,macleod_ratio_short=w**(b_sf-b_tau/2),
                      macleod_ratio_long=w**b_sf)).to_csv(OUT/'structure_function_band_ratio.csv',index=False)


def drw_growth():
    fit=pd.read_csv(OUT/'structure_function_object_fits.csv').query('sample == "fixed" and model == "drw_short"').set_index('band')
    rows=[]
    with np.load(OUT/'ensemble_sf_covariance.npz') as f:
        for band in esf.BANDS:
            kernel=f[f'lag_nodes_{band}'],f[f'lag_weights_{band}']
            lag=fsf.CENTERS[[int(np.argmin(abs(fsf.CENTERS-2371))),kernel[0].shape[1]-1]]
            tau=fit.timescale_or_slope[band]
            sf=fsf._object_sampled_models(kernel)[0](lag,fit.amplitude[band],tau)
            rows.append(dict(band=band,anchor_lag_days=lag[0],endpoint_lag_days=lag[1],tau_days=tau,
                             drw_anchor_mag=sf[0],drw_endpoint_mag=sf[1],growth_percent=100*(sf[1]/sf[0]-1),
                             endpoint_over_tau=lag[1]/tau))
    pd.DataFrame(rows).to_csv(OUT/'structure_function_drw_growth.csv',index=False)


def group_trends():
    pairs={'lowmbh':('lowmbh_lowedd','lowmbh_highedd'),'highmbh':('highmbh_lowedd','highmbh_highedd'),
           'lowedd':('lowmbh_lowedd','highmbh_lowedd'),'highedd':('lowmbh_highedd','highmbh_highedd')}
    matched=pd.read_csv(OUT/'structure_function_matched_mbh_edd.csv').rename(columns=lambda c:c.replace('sf_raw','sf'))
    narrow=pd.read_csv(OUT/'structure_function_narrowing_curves.csv').query('selection == "redshift_060_090"')
    half=lambda s:(s.sf_hi_mag-s.sf_lo_mag)/2
    rows=[]
    for selection,d in [('all',matched),('redshift_060_090',narrow)]:
        for band in esf.BANDS:
            for fixed,(lo,hi) in pairs.items():
                x,y=(d[d.band.eq(band)&d['sample'].eq(s)].set_index('lag_center_days') for s in (lo,hi))
                if x.empty or y.empty:
                    continue
                k=x.index[np.argmin(abs(x.index-2371))]
                z=[s.loc[k].get('z_med',np.nan) for s in (x,y)]
                rows.append(dict(band=band,selection=selection,fixed=fixed,lo_group=lo,hi_group=hi,lag_center_days=k,
                                 n_lo=x.n_objects[k],n_hi=y.n_objects[k],sf_lo_mag=x.sf_mag[k],sf_hi_mag=y.sf_mag[k],
                                 diff_hi_minus_lo=y.sf_mag[k]-x.sf_mag[k],diff_err=np.hypot(half(x.loc[k]),half(y.loc[k])),
                                 n_bins_hi_below=int((y.sf_mag<x.sf_mag).sum()),n_bins=len(x),z_med_lo=z[0],z_med_hi=z[1],
                                 macleod_ratio=((1+z[1])/(1+z[0]))**-fsf.POPDRW_COEF['sfinf'][1]))
    pd.DataFrame(rows).to_csv(OUT/'structure_function_group_trends.csv',index=False)


def noise_fraction():
    from plate_completeness import load_completeness_lightcurves
    zmap=pd.read_parquet(OUT/'plate_completeness_sample.parquet').drop_duplicates('OBJID').assign(
        OBJID=lambda d:d.OBJID.astype(str)).set_index('OBJID').z
    rows=[]
    with np.load(OUT/'structure_function_sampling_covariance.npz') as f:
        for b in esf.BANDS:
            ids,fid=f[f'fiducial_ids_{b}'],f[f'fiducial_acc_{b}']
            nb=fid.shape[1]
            pairs=f[f'fiducial_asymmetry_acc_{b}'][:,:2,0,:nb,0].sum(0)
            nt=esf.clean_nightly(load_completeness_lightcurves(ids))[0].query('band == @b')
            a=esf.accumulate(nt,ids,zmap,bands=[b],resid=True)[:,0,:nb]
            n=a[...,0]
            dm2,noise,cal=((x/n).mean(0) for x in (a[...,1],a[...,2]-a[...,4],a[...,4]))
            sf2=dm2-noise-cal
            np.testing.assert_allclose(sf2,esf.ensemble_second_moment(fid),rtol=1e-9)
            rows+=[dict(band=b,bin=k,lag_center_days=esf.CENTERS[k],n_objects=len(ids),plate_pair_fraction=1-pairs[1,k]/pairs[0,k],
                        mean_square_mag2=dm2[k],noise_mag2=noise[k],calibration_mag2=cal[k],sf2_mag2=sf2[k],sf_mag=np.sqrt(sf2[k]),
                        noise_over_sf2=noise[k]/sf2[k],calibration_over_sf2=cal[k]/sf2[k])
                   for k in range(nb)]
    pd.DataFrame(rows).to_csv(OUT/'structure_function_noise_fraction.csv',index=False)


def star_lcs(lo, hi):
    import ccd_standard_star_errors as cse
    lc = cse.load_star_lcs(cse.excess_curves())
    st = cse.star_stats(lc)
    st = st[st.band.isin(esf.BANDS) & st.medmag.between(st.band.map(lo), st.band.map(hi))]
    return lc.merge(st[['survey','band','OBJID']]).rename(columns={'err_used':'magerr'})


def star_floor(lo, hi, n_boot=2000, seed=930):
    lc = star_lcs(lo, hi)
    rng, out = np.random.default_rng(seed), {}
    for sv, s in lc.groupby('survey'):
        nt = esf.reject_outliers(esf.condense_nightly(s))[0]
        ids = np.sort(nt.OBJID.unique())
        acc = esf.accumulate(nt, ids, pd.Series(0., index=ids)).sum(axis=2)
        for bi, b in enumerate(esf.BANDS):
            use = acc[:, bi, 0] > 0
            n, d = acc[use, bi, 0], acc[use, bi, 1]-acc[use, bi, 2]
            draw = rng.multinomial(len(n), np.full(len(n), 1/len(n)), size=n_boot)
            out[sv, b] = d.sum()/n.sum(), draw@d/(draw@n)
    return out


def fit_range():
    from plate_completeness import load_completeness_lightcurves
    C, ccd = fsf.CENTERS, ['sdss', 'ps1', 'ztf']
    subs = ['lowmbh_lowedd', 'lowmbh_highedd', 'highmbh_lowedd', 'highmbh_highedd']
    with np.load(OUT/'ensemble_sf_covariance.npz') as f:
        reps = {b: f[f'replicates_{b}'] for b in esf.BANDS}
    with np.load(OUT/'structure_function_sampling_covariance.npz') as f:
        pairs = {b: f[f'fiducial_asymmetry_acc_{b}'][:, :2, 0, :, 0] for b in esf.BANDS}
    fid, sub = {}, {}
    for index, (band, name, a, prop, kernel) in enumerate(samples()):
        if name == 'fiducial':
            fid[band] = a, reps[band], kernel, prop, index
        elif name in subs:
            r = esf.object_bootstrap(a, n_boot=3000, seed=3300+10*esf.BANDS.index(band)+subs.index(name))[1]
            sub[band, name] = a, np.sqrt(np.maximum(r, 0)), kernel
    ref = pd.read_parquet(OUT/'plate_completeness_sample.parquet').astype({'OBJID': str}).set_index(['band', 'OBJID']).reference_mag
    mags = {b: ref[b].reindex(fid[b][3].index) for b in esf.BANDS}
    stars = star_floor({b: m.min() for b, m in mags.items()}, {b: m.max() for b, m in mags.items()})
    noise = pd.read_csv(OUT/'structure_function_noise_fraction.csv').set_index(['band', 'bin']).noise_over_sf2
    rows = []
    for b in esf.BANDS:
        a, r, kernel, prop, _ = fid[b]
        nb, ids = a.shape[1], prop.index.to_numpy(str)
        nt = esf.clean_nightly(load_completeness_lightcurves(ids))[0].query('band == @b')
        coef = []
        for s in ccd:
            acc = esf.accumulate(nt.assign(mag=0., sig=nt.survey.eq(s).astype(float)), ids, prop.z, bands=[b])[:, 0, :nb]
            np.testing.assert_array_equal(acc[..., 0], a[..., 0])
            coef.append(-esf.ensemble_second_moment(acc)/2)
        floor = np.array([stars[s, b][0] for s in ccd])@coef
        floor_reps = np.stack([stars[s, b][1] for s in ccd], axis=1)@coef
        sf, sigma = np.sqrt(esf.ensemble_second_moment(a)), r.std(axis=0, ddof=1)
        shift, n = sf-np.sqrt(sf**2-floor), pairs[b][..., :nb]
        rows += [dict(band=b, bin=k, lag_center_days=C[k], n_objects=len(a), sf_mag=sf[k], sf_err_mag=sigma[k],
                      floor_mag2=floor[k], floor_err_mag2=floor_reps[:, k].std(ddof=1), shift_mag=shift[k],
                      shift_over_sigma=shift[k]/sigma[k], ccd_pair_fraction=n[:, 1, k].sum()/n[:, 0, k].sum(),
                      ccd_pair_fraction_object=(n[:, 1, k]/n[:, 0, k]).mean(), noise_over_sf2=noise[b, k],
                      star_mag_lo=mags[b].min(), star_mag_hi=mags[b].max(),
                      **{f'star_floor_{s}_mag2': stars[s, b][0] for s in ccd},
                      **{f'star_floor_{s}_err_mag2': stars[s, b][1].std(ddof=1) for s in ccd}) for k in range(nb)]
    fl = pd.DataFrame(rows)
    wide = lambda c: fl.pivot(index='bin', columns='band', values=c)
    end = int(np.flatnonzero(wide('ccd_pair_fraction_object').ge(.99).all(axis=1).to_numpy())[-1])
    def start(col, bands=esf.BANDS):
        fail = np.flatnonzero(~wide(col)[list(bands)].abs().lt(1).all(axis=1).to_numpy()[:end+1])
        return C[fail[-1]+1 if len(fail) else 0]
    grid, out = psd_grid(), []
    last = lambda fit: next(x for x in reversed(fit[0]) if x['model'] == 'drw_extrapolation')
    for b in esf.BANDS:
        a, r, kernel, _, index = fid[b]
        drw, _, jac = fsf._object_sampled_models(kernel)
        y, dt = np.sqrt(esf.ensemble_second_moment(a)), C[:a.shape[1]]
        def record(model, i, j, n, p, e, rc, ex, err, **extra):
            m = np.sqrt(drw(dt, *p[:2])**2+(p[2] if len(p) > 2 else 0))
            out.append(dict(band=b, model=model, start_days=C[i], end_days=C[j], n_bins=n, dof=n-len(p),
                            sf_inf_mag=p[0], sf_inf_err_mag=e[0], tau_days=p[1], tau_err_days=e[1],
                            **dict(zip(['floor_mag2', 'floor_err_mag2'], [*p[2:], *e[2:]])), reduced_chi2=rc,
                            end_resid_mag2=y[j]**2-m[j]**2, final_lag_days=dt[-1], final_excess_mag=ex,
                            final_excess_err_mag=err, final_excess_sigma=ex/err, **extra))
        for i in (3, 4, 5, 6, 7):
            psd = {x['model']: x for x in fit_psd(a, kernel, grid, 928+index, C[i])[0]}
            same = dict(same_bin_drw_reduced_chi2=psd['drw']['reduced_chi2'],
                        same_bin_free_reduced_chi2=psd['bending_psd']['reduced_chi2'],
                        same_bin_alpha=psd['bending_psd']['alpha'], same_bin_delta_chi2=psd['bending_psd']['delta_chi2_from_drw'])
            for j in (9, 10):
                fit = fsf._object_fit(a, r, 'fixed', b, kernel, (C[i], C[j]))
                if fit is None:
                    continue
                s, e = fit[0][0], last(fit)
                for name in subs:
                    q = fsf._object_fit(*sub[b, name][:2], name, b, sub[b, name][2], (C[i], C[j]))
                    x = last(q) if q else dict(excess_mag=np.nan, excess_err_mag=np.nan)
                    same[f'{name}_excess_mag'], same[f'{name}_sigma'] = x['excess_mag'], x['excess_mag']/x['excess_err_mag']
                record('drw', i, j, s['n_bins'], fit[2], [s['amplitude_err'], s['timescale_or_slope_err']],
                       s['reduced_chi2'], e['excess_mag'], e['excess_err_mag'], **same)
        cov = fsf._object_fit(a, r, 'fixed', b, kernel)[3]
        model = lambda x, A, t, c: np.sqrt(drw(x, A, t)**2+c)
        for j in (9, 10):
            m = dt <= C[j]
            p, e, pc, rc = min((fsf.fit(model, dt[m], y[m], cov[np.ix_(m, m)], [y[m].max(), t, 0.], ([1e-3, 1., -5e-3], [3., 1e6, 5e-3]))
                                for t in fsf.GLS_TAU0), key=lambda q: q[3])
            M = model(dt, *p)
            J = np.column_stack([jac(dt, *p[:2])*(drw(dt, *p[:2])/M)[:, None], .5/M])
            err = np.sqrt(fsf._extrapolation_cov(cov, m, J, pc, rc)[-1, -1])
            record('drw_floor', 0, j, int(m.sum()), p, e, rc, y[-1]-M[-1], err)
    fits = pd.DataFrame(out)
    fits.to_csv(OUT/'structure_function_fit_range.csv', index=False)
    q = fits[fits.model.eq('drw_floor') & fits.end_days.eq(C[end])].set_index('band')
    c, ce = (q.loc[fl.band, k].to_numpy() for k in ['floor_mag2', 'floor_err_mag2'])
    shift = lambda x: fl.sf_mag-np.sqrt(fl.sf_mag**2-x)
    fl = fl.assign(quasar_floor_mag2=c, quasar_floor_err_mag2=ce, quasar_shift_mag=shift(c), quasar_shift_over_sigma=shift(c)/fl.sf_err_mag,
                   quasar_shift_over_sigma_lo=shift(c-ce)/fl.sf_err_mag, quasar_shift_over_sigma_hi=shift(c+ce)/fl.sf_err_mag)
    fl.assign(criterion_start_days=start('shift_over_sigma'), criterion_end_days=C[end],
              quasar_criterion_start_days=fl.band.map({b: start('quasar_shift_over_sigma', b) for b in esf.BANDS}),
              quasar_criterion_start_joint_days=start('quasar_shift_over_sigma')).to_csv(OUT/'structure_function_fit_range_floor.csv', index=False)


REJECT_SIGMA = (3., 4., 5., 7., 10., np.inf)


def deviations(nightly):
    n = esf.segment_deviations(nightly).sort_values([*esf.SEGMENT, 'night'], ignore_index=True)
    return n.assign(rob=np.maximum(n.mad, n.err), noisy=n.err >= n.mad,
                    nseg=n.groupby(esf.SEGMENT, observed=True, sort=False).mag.transform('size'))


def night_rows(n, sample):
    rows = []
    for (b, s), d in [*n.groupby(['band', 'survey']), *(((b, 'all'), d) for b, d in n.groupby('band'))]:
        d = d.sort_values([*esf.SEGMENT, 'night'])
        res, rob = d.res.to_numpy(), d.rob.to_numpy()
        key = (d.night+1e7*d.groupby(esf.SEGMENT).ngroup()).to_numpy(float)
        night = d.night.to_numpy()
        seg = d.drop_duplicates(esf.SEGMENT)
        base = dict(sample=sample, band=b, survey=s, n_objects=d.OBJID.nunique(), n_nights=len(d),
                    segment_nights_median=seg.nseg.median(), nights_in_3plus_fraction=(d.nseg >= 3).mean(),
                    noise_dominated_segment_fraction=seg.noisy.mean(), max_abs_dev_sigma=np.abs(res/rob).max())
        for t in REJECT_SIGMA:
            far = np.abs(res) > t*rob
            r, c = np.flatnonzero(far), np.concatenate([[0], np.cumsum(far)])
            lo, hi = np.searchsorted(key, key[r]-30, 'left'), np.searchsorted(key, key[r]+30, 'right')
            near, bad, x = hi-lo-1, c[hi]-c[lo]-1, pd.Series(np.abs(res[r]))
            g = pd.Series(far).groupby(night)
            k, m, w = g.transform('sum').to_numpy(), g.transform('size').to_numpy(), g.sum().idxmax()
            on = (night == w) & (len(r) > 0)
            rows.append(dict(base, threshold_sigma=t, n_rejected=len(r), rejected_fraction=len(r)/len(d),
                             n_objects_affected=d.OBJID.iloc[r].nunique(), rejected_abs_dev_mag_median=x.median(),
                             rejected_abs_dev_mag_q90=x.quantile(.9), rejected_abs_dev_mag_max=x.max(),
                             rejected_abs_dev_sigma_median=(x/rob[r]).median(), rejected_fainter_fraction=pd.Series(res[r] > 0).mean(),
                             n_isolated=int(((near > 0) & (bad == 0)).sum()), n_clustered=int((bad > 0).sum()),
                             n_no_neighbour=int((near == 0).sum()),
                             rejected_coincident_fraction=pd.Series(((m >= 5) & (k-1 >= .2*(m-1)))[r]).mean(),
                             worst_night=w if len(r) else np.nan, worst_night_rejected=far[on].sum(),
                             worst_night_objects=on.sum(), worst_night_median_res_mag=pd.Series(res[on]).median()))
    return rows


def rejection():
    from plate_completeness import load_completeness_lightcurves
    member = pd.read_parquet(OUT/'structure_function_sampling_membership.parquet').query('fiducial')
    ids = {b: np.sort(member[member.band.eq(b)].OBJID.to_numpy(str)) for b in esf.BANDS}
    with np.load(OUT/'ensemble_sf_covariance.npz') as f:
        reps = {b: f[f'replicates_{b}'] for b in esf.BANDS}
        for b in esf.BANDS:
            np.testing.assert_array_equal(f[f'objids_{b}'].astype(str), ids[b])
    cat = pd.read_parquet(OUT/'plate_completeness_sample.parquet').astype({'OBJID': str})
    prop, ref = cat.drop_duplicates('OBJID').set_index('OBJID'), cat.set_index(['band', 'OBJID']).reference_mag
    nightly = esf.condense_nightly(load_completeness_lightcurves(np.union1d(*ids.values())))
    q = deviations(nightly)
    order = lambda x: x[list(nightly.columns)].sort_values([*esf.SEGMENT, 'night'], ignore_index=True)
    pd.testing.assert_frame_equal(order(esf.reject_outliers(nightly)[0]), order(q[np.abs(q.res) <= esf.CLIP_SIGMA*q.rob]))
    q = pd.concat([q[q.band.eq(b) & q.OBJID.isin(ids[b])] for b in esf.BANDS])
    mags = {b: ref[b].reindex(ids[b]) for b in esf.BANDS}
    lo, hi = ({b: f(m) for b, m in mags.items()} for f in (np.min, np.max))
    stars = pd.concat([deviations(esf.condense_nightly(s)) for _, s in star_lcs(lo, hi).groupby('survey')])
    pd.DataFrame(night_rows(q, 'quasar')+night_rows(stars, 'star')).to_csv(OUT/'structure_function_rejection_nights.csv', index=False)
    k0, rows = int(np.argmin(abs(esf.CENTERS-2371))), []
    cols = ['variant', 'threshold_sigma', 'n_restored', 'band', 'bin', 'lag_center_days', 'n_objects', 'n_pairs', 'sf_mag',
            'sf_lo_mag', 'sf_hi_mag', 'sf_err_mag', 'rise_mag', 'rise_lo_mag', 'rise_hi_mag', 'rise_bootstrap_sigma']
    keep = lambda d, t: d[np.abs(d.res) <= t*d.rob]
    for bi, b in enumerate(esf.BANDS):
        d, nb = q[q.band.eq(b)], reps[b].shape[1]
        z, lum = (prop[c].reindex(ids[b]).to_numpy(float) for c in ('z', 'loglbol'))
        def sf(x, variant, t=esf.CLIP_SIGMA, n=np.nan):
            s, r = fsf._object_summary(esf.accumulate(x, ids[b], prop.z, bands=[b], resid=True)[:, 0, :nb], z, lum, seed=1927+bi)
            dr = r-r[:, [k0]]
            rows.append(s.assign(variant=variant, threshold_sigma=t, n_restored=n, band=b, bin=np.arange(nb),
                                 rise_mag=s.sf_mag-s.sf_mag[k0], rise_lo_mag=np.percentile(dr, 16, 0),
                                 rise_hi_mag=np.percentile(dr, 84, 0), rise_bootstrap_sigma=dr.std(0, ddof=1))[cols])
            return r
        for t in REJECT_SIGMA:
            r = sf(keep(d, t), 'threshold', t)
            if t == esf.CLIP_SIGMA:
                np.testing.assert_allclose(r, reps[b], rtol=1e-9, atol=1e-12)
        out = d[np.abs(d.res) > esf.CLIP_SIGMA*d.rob].sort_values('res', key=np.abs, ascending=False)
        for n in (1, 3, 10, 30, 100):
            sf(pd.concat([keep(d, esf.CLIP_SIGMA), out[:n]]), 'restore_largest', n=n)
        for v, x in out.groupby('survey'):
            sf(pd.concat([keep(d, esf.CLIP_SIGMA), x]), f'restore_{v}', n=len(x))
    table = pd.concat(rows, ignore_index=True)
    adopted = table[table.variant.eq('threshold') & table.threshold_sigma.eq(esf.CLIP_SIGMA)]
    fixed = pd.read_csv(OUT/'structure_function_fixed_population.csv').query("sample == 'fixed'")
    rises = pd.read_csv(OUT/'structure_function_object_rises.csv').query("sample == 'fixed'")
    np.testing.assert_allclose(adopted[cols[8:11]], fixed[cols[8:11]], rtol=1e-9)
    np.testing.assert_allclose(adopted[adopted.bin > k0][cols[12:]], rises[cols[12:]], rtol=1e-9)
    table.to_csv(OUT/'structure_function_rejection_sensitivity.csv', index=False)


def population_drw_grid(kernel, amplitude, tau, logscale, scatter=True, order=16):
    nodes, weights = kernel[:2]
    sa, st = fsf.POPDRW_SIG if scatter else (0., 0.)
    va, vt, cv = .64*sa**2+.16*st**2, .16*sa**2+.64*st**2, .32*(st**2-sa**2)
    x, w = np.polynomial.hermite.hermgauss(order if scatter else 1)
    multiplier = np.exp(2*cv*np.log(10)**2+np.sqrt(2*vt)*np.log(10)*x)
    normalization = amplitude**2*np.exp(2*va*np.log(10)**2)
    values = []
    for scale in np.atleast_1d(logscale):
        q = nodes[..., None]/(tau[:, None, None, None]*10**scale*multiplier)
        v = (-np.expm1(-q)*w/np.sqrt(np.pi)).sum(axis=-1)
        values.append(normalization[:, None]*(v*weights).sum(axis=2))
    return np.stack(values, axis=-1)


def population_drw_fit(grid, logscale, values, cov, short):
    curve = CubicSpline(logscale, grid, axis=-1)
    inv = np.linalg.inv(cov[np.ix_(short, short)])
    y = values[short]
    def profile(t):
        h = np.sqrt(curve(t))[short]
        amplitude = (h@inv@y)/(h@inv@h)
        r = y-amplitude*h
        return float(r@inv@r), amplitude
    loss = np.array([profile(t)[0] for t in logscale])
    i = int(loss.argmin())
    if i in (0, len(logscale)-1):
        raise ValueError('Population DRW timescale fit reaches the evaluation boundary')
    result = minimize_scalar(lambda t: profile(t)[0], bounds=logscale[[i-1, i+1]],
                             method='bounded', options={'xatol':1e-10})
    if not result.success:
        raise RuntimeError(result.message)
    t = result.x
    x2, amplitude = profile(t)
    if amplitude <= 0:
        raise ValueError('Nonpositive population DRW amplitude')
    shape = np.sqrt(curve(t))
    jac = np.column_stack([shape, amplitude*curve(t, 1)/(2*shape)])
    pc = np.linalg.inv(jac[short].T@inv@jac[short])
    return np.array([amplitude, t]), amplitude*shape, pc, jac, x2/(short.sum()-2)


def population_properties():
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    x = pd.read_fwf(OUT/'macleod2012_southcat.dat',colspecs=[(0,7),(9,19),(22,31),(40,47),(49,56),(58,64)],
                    names=['DBID','catalog_ra','catalog_dec','Mi_z0','Mi_z2','catalog_z'])
    archive = np.load(OUT/'ensemble_sf_covariance.npz')
    ids = np.unique(np.concatenate([archive[f'objids_{b}'].astype(str) for b in esf.BANDS]))
    cat = pd.read_parquet(OUT/'S82/Catalog.parquet').assign(OBJID=lambda q:q.objectId.astype(str)).set_index('OBJID').loc[ids]
    j,separation,_ = SkyCoord(cat.RA.to_numpy()*u.deg,cat.DEC.to_numpy()*u.deg).match_to_catalog_sky(
        SkyCoord(x.catalog_ra.to_numpy()*u.deg,x.catalog_dec.to_numpy()*u.deg))
    q = x.iloc[j].reset_index(drop=True)
    q['OBJID'],q['separation_arcsec'],q['z_current'] = ids,separation.arcsec,cat.Z_DR16Q.to_numpy()
    q['matched'],q['source'] = separation.arcsec<1,'MacLeod2012_southcat'
    for column in ['Mi_z0','Mi_z2']:
        q[column] = q[column].where(q.matched & q[column].lt(-10))
    q['DBID'] = q.DBID.where(q.matched).astype('Int64')
    q.to_csv(OUT/'structure_function_population_properties.csv',index=False)
    return q.set_index('OBJID')


def population_drw_test():
    properties = population_properties()
    archive = np.load(OUT/'ensemble_sf_covariance.npz')
    logscale = np.linspace(-3, 3, 181)
    fits, curves, objects = [], [], []
    for bi, (band, name, a, prop, kernel) in enumerate(q for q in samples() if q[1]=='fiducial'):
        matched = properties.reindex(prop.index)[['Mi_z0','Mi_z2']].notna().all(axis=1).to_numpy()
        relation = [('macleod_population','Mi_z0',fsf.POPDRW_COEF),
                    ('suberlak_population','Mi_z2',{'sfinf':(-.476,-.479,.118,.118),'tau':(2.597,.17,.035,.141)})]
        models = [('single_full',np.ones(len(a)),np.full(len(a),300.),False,np.ones(len(a),bool)),
                  ('single',np.ones(matched.sum()),np.full(matched.sum(),300.),False,matched)]
        for label,column,coefficients in relation:
            pp = prop.loc[matched]
            mi = properties.reindex(pp.index)[column].to_numpy(float)
            predictors = np.column_stack([np.ones(len(pp)),np.log10(fsf.POPDRW_LAM_OBS[band]/(1+pp.z)/4000),
                                          mi+23,pp.LOGMBH-9])
            amp0,tau0 = [10**(predictors@coefficients[c]) for c in ['sfinf','tau']]
            models.append((label,amp0,tau0,True,matched))
            objects.extend(dict(OBJID=o,band=band,model=label,Mi=m,magnitude_convention=column,
                logmbh=b,z=z,sfinf_mag=s,tau_days=t)
                for o,m,b,z,s,t in zip(pp.index,mi,pp.LOGMBH,pp.z,amp0,tau0))
        moments = esf.object_second_moments(a)
        dt = fsf.CENTERS[:a.shape[1]]
        short = fsf.plateau_bins(dt)
        long = dt > fsf.LONG_MIN_D
        nboot = len(archive[f'replicates_{band}'])
        ids = np.random.default_rng(1927+bi).integers(0,len(a),(nboot,len(a)))
        counts = np.zeros((nboot,len(a)))
        np.add.at(counts,(np.arange(nboot)[:,None],ids),1)
        np.testing.assert_allclose(np.sqrt(counts@moments/len(a)),archive[f'replicates_{band}'],rtol=1e-12,atol=1e-12)
        for label,amplitude,tau,scatter,mask in models:
            local_kernel = tuple(k[mask] for k in kernel)
            n = int(mask.sum())
            y = np.sqrt(moments[mask].mean(axis=0))
            count = counts[:,mask]
            count /= count.sum(axis=1)[:,None]
            reps = np.sqrt(count@moments[mask])
            sd = reps.std(axis=0,ddof=1)
            corr,_,_ = fsf.ledoit_wolf(reps/sd)
            cov = corr*np.outer(sd,sd)
            table = population_drw_grid(local_kernel,amplitude,tau,logscale,scatter)
            mean = table.mean(axis=0)
            unscaled = np.sqrt(mean[:, np.argmin(abs(logscale))])
            par, predicted, pc, jac, rc = population_drw_fit(mean,logscale,y,cov,short)
            exact = population_drw_grid(local_kernel,amplitude,tau,[par[1]],scatter,32).mean(axis=0)[:,0]
            integration_error = np.max(abs(par[0]*np.sqrt(exact)-predicted))
            if integration_error > 1e-6:
                raise ValueError(f'Population quadrature/interpolation error {integration_error}')
            print(f'{band} {label}: short scales {par}, chi2/df {rc:.3f}, endpoint {predicted[-1]:.6f} versus {y[-1]:.6f}',flush=True)
            weighted = (count@table.reshape(n,-1)).reshape(nboot,*mean.shape)
            parameters, residuals = [], []
            for r, (data, model) in enumerate(zip(reps,weighted)):
                p, fitted, _, _, _ = population_drw_fit(model,logscale,data,cov,short)
                parameters.append(p)
                residuals.append(data-fitted)
            residuals, parameters = np.asarray(residuals), np.asarray(parameters)
            empirical = np.cov(residuals,rowvar=False)
            extra = (max(rc,1)-1)*jac@pc@jac.T
            residual = y-predicted
            summary = dict(band=band,model=label,n_objects=n,n_parent_objects=len(a),n_bootstrap=nboot,n_fit_bins=int(short.sum()),
                amplitude_scale=par[0],timescale_scale=10**par[1],reduced_chi2_short=rc,
                integration_error_mag=integration_error,fit_min_days=dt[short].min(),fit_max_days=dt[short].max(),
                chi2_short=rc*(short.sum()-2),dof_short=int(short.sum()-2),n_bootstrap_failed=0,n_bootstrap_boundary=0)
            for inflation, S in [('unscaled',empirical),('fit_scaled',empirical+extra)]:
                r = residual[long]
                x2 = float(r@np.linalg.solve(S[np.ix_(long,long)],r))
                probability = chi2.sf(x2,int(long.sum()))
                fits.append(dict(**summary,uncertainty=inflation,joint_chi2=x2,n_long_bins=int(long.sum()),
                    gaussian_p=probability,gaussian_sigma=norm.isf(probability/2),final_prediction_mag=predicted[-1],
                    final_excess_mag=residual[-1],final_excess_err_mag=np.sqrt(S[-1,-1]),
                    amplitude_lo95=np.percentile(parameters[:,0],2.5),amplitude_hi95=np.percentile(parameters[:,0],97.5),
                    timescale_lo95=10**np.percentile(parameters[:,1],2.5),timescale_hi95=10**np.percentile(parameters[:,1],97.5)))
                for k in range(len(y)):
                    curves.append(dict(band=band,model=label,uncertainty=inflation,lag_days=dt[k],in_fit=bool(short[k]),
                        observed_mag=y[k],prediction_mag=predicted[k],unscaled_prediction_mag=unscaled[k],
                        residual_mag=residual[k],residual_sd_mag=np.sqrt(S[k,k])))
            np.savez_compressed(esf.cache_path(f'model_checks/population_drw_{band}_{label}.npz'),parameters=parameters,residuals=residuals,
                                covariance=empirical,extra_covariance=extra)
            print(f'{band} {label}: {par}, short chi2/df {rc:.3f}, long sigma {fits[-1]["gaussian_sigma"]:.3f}',flush=True)
    pd.DataFrame(fits).to_csv(OUT/'structure_function_population_fits.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'structure_function_population_curves.csv',index=False)
    pd.DataFrame(objects).to_csv(OUT/'structure_function_population_objects.csv',index=False)


def fiducial_nightly():
    from plate_completeness import load_completeness_lightcurves
    selected = [x for x in samples() if x[1] == 'fiducial']
    nightly = esf.clean_nightly(load_completeness_lightcurves(np.unique(np.concatenate([x[3].index for x in selected]))))[0]
    for band, _, base, p, _ in selected:
        n = nightly[nightly.band.eq(band) & nightly.OBJID.isin(p.index)].reset_index(drop=True)
        run = lambda x, ids=p.index.to_numpy(): esf.accumulate(x, ids, p.z, bands=[band], edges=esf.EDGES[:base.shape[1]+1], resid=True)[:, 0]
        real = run(n)
        np.testing.assert_allclose(real[..., :3], base[..., :3], rtol=1e-10, atol=1e-8)
        yield band, base, p, n, run, real


def replace_plate_epochs(n, rng, variant):
    sss, m, s2, fv, survey = n.sss.to_numpy(bool), n.mag.to_numpy(), n.sig.to_numpy()**2, n.fv0.to_numpy(), n.survey.to_numpy()
    out = m.copy()
    for rows in n.groupby('OBJID', sort=False).indices.values():
        plate, ccd = rows[sss[rows]], rows[~sss[rows]]
        if variant == 'ccd_median':
            source, var = np.full(len(plate), np.median(m[ccd])), s2[plate]+fv[plate]
        else:
            _, inv, count = np.unique(survey[ccd], return_inverse=True, return_counts=True)
            pick = rng.choice(ccd, len(plate), replace=False, p=None if variant == 'ccd_night' else 1/(len(count)*count[inv]))
            source, var = m[pick], s2[plate]+fv[plate]-s2[pick]
        out[plate] = source+np.sqrt(np.maximum(var, 0))*rng.standard_normal(len(plate))
    return n.assign(mag=out)


def plate_replacement(n_rep=200, n_boot=5000):
    fixed = pd.read_csv(OUT/'structure_function_fixed_population.csv').query('sample == "fixed"')
    with np.load(OUT/'ensemble_sf_covariance.npz') as f:
        adopted = {b: f[f'replicates_{b}'] for b in esf.BANDS}
    rows, variants = [], ['ccd_night', 'ccd_survey', 'ccd_median']
    for bi, (band, base, p, n, run, real) in enumerate(fiducial_nightly()):
        rng = np.random.default_rng(10050+bi)
        draws = {v: np.stack([run(replace_plate_epochs(n, rng, v)) for _ in range(n_rep)]) for v in variants}
        fixed_slots = [0, 2, 3, 4, 5]
        for d in draws.values():
            np.testing.assert_array_equal(d[..., fixed_slots], np.broadcast_to(real[..., fixed_slots], d[..., fixed_slots].shape))
        stack = np.stack([real, *(d.mean(axis=0) for d in draws.values())], axis=1)
        bounds, boot = (np.sqrt(np.maximum(x, 0)) for x in esf.object_bootstrap(stack, n_boot=n_boot, seed=1927+bi))
        sf = np.sqrt(np.maximum(esf.ensemble_second_moment(stack), 0))
        tab = fixed[fixed.band.eq(band)]
        np.testing.assert_allclose(boot[:, 0], adopted[band], rtol=1e-12)
        np.testing.assert_allclose(np.stack([sf[0], bounds[0, 0], bounds[2, 0]]), tab[['sf_mag', 'sf_lo_mag', 'sf_hi_mag']].T, rtol=1e-10)
        frac = (real[..., 3]/real[..., 0]).mean(axis=0)
        for vi, (v, d) in enumerate(draws.items(), 1):
            null = np.sqrt(np.maximum([esf.ensemble_second_moment(x) for x in d], 0))
            q = np.percentile(null, [2.5, 16, 84, 97.5], axis=0)
            diff = boot[:, 0]-boot[:, vi]
            dq = np.percentile(diff, [16, 84], axis=0)
            for k in range(base.shape[1]):
                rows.append(dict(band=band, variant=v, lag_center_days=fsf.CENTERS[k], n_objects=len(p), n_realizations=n_rep,
                    plate_pair_fraction=frac[k], sf_mag=sf[0, k], sf_lo_mag=bounds[0, 0, k], sf_hi_mag=bounds[2, 0, k],
                    null_mean_mag=null[:, k].mean(), null_sd_mag=null[:, k].std(ddof=1), null_p2_5_mag=q[0, k],
                    null_p16_mag=q[1, k], null_p84_mag=q[2, k], null_p97_5_mag=q[3, k],
                    null_exceed_p=((null[:, k] >= sf[0, k]).sum()+1)/(n_rep+1) if frac[k] else np.nan,
                    conditional_sigma=(sf[0, k]-null[:, k].mean())/null[:, k].std(ddof=1) if frac[k] else np.nan,
                    null_expected_mag=sf[vi, k], excess_sf2_mag2=sf[0, k]**2-sf[vi, k]**2,
                    difference_mag=sf[0, k]-sf[vi, k], difference_lo_mag=dq[0, k], difference_hi_mag=dq[1, k],
                    bootstrap_sigma=(sf[0, k]-sf[vi, k])/diff[:, k].std(ddof=1) if frac[k] else np.nan))
        out = pd.DataFrame(rows)
        print(out[out.band.eq(band) & (out.lag_center_days > fsf.LONG_MIN_D)][['variant', 'lag_center_days', 'sf_mag', 'null_mean_mag',
              'null_p97_5_mag', 'conditional_sigma', 'bootstrap_sigma']].to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(OUT/'structure_function_plate_replacement.csv', index=False)


def main():
    grid=psd_grid()
    fits,profiles,curves,narrow,properties,narrow_curves,influences=[],[],[],[],[],[],[]
    for index,(band,name,a,prop,kernel) in enumerate(samples()):
        print(f'{band} {name}: {len(a)} objects',flush=True)
        rows,pp,cc=fit_psd(a,kernel,grid,928+index)
        for row in rows:
            row.update(band=band,sample=name,n_objects=len(a))
        fits.extend(rows)
        profiles.extend([dict(band=band,sample=name,**p) for p in pp])
        curves.append(cc.assign(band=band,sample=name))
        nr,pr,nc=nested_checks(a,prop,2928+index)
        narrow.extend([dict(band=band,sample=name,**p) for p in nr])
        properties.extend([dict(band=band,sample=name,**p) for p in pr])
        narrow_curves.extend([dict(band=band,sample=name,**p) for p in nc])
        if band=='r' and name in ['grid_0_2','grid_1_2']:
            influences.extend([dict(band=band,sample=name,**p) for p in influence(a,prop)])
    pd.DataFrame(fits).to_csv(OUT/'structure_function_psd_fits.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUT/'structure_function_psd_profiles.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(OUT/'structure_function_psd_curves.csv',index=False)
    pd.DataFrame(narrow).to_csv(OUT/'structure_function_narrowing_checks.csv',index=False)
    pd.DataFrame(properties).to_csv(OUT/'structure_function_narrowing_properties.csv',index=False)
    pd.DataFrame(narrow_curves).to_csv(OUT/'structure_function_narrowing_curves.csv',index=False)
    pd.DataFrame(influences).to_csv(OUT/'structure_function_high_edd_influence.csv',index=False)


if __name__ == '__main__':
    esf.check_flags(__file__)
    if '--lowmass-influence' in sys.argv:
        lowmass_influence()
    elif '--band-ratio' in sys.argv:
        band_ratio()
    elif '--drw-growth' in sys.argv:
        drw_growth()
    elif '--group-trends' in sys.argv:
        group_trends()
    elif '--noise-fraction' in sys.argv:
        noise_fraction()
    elif '--fit-range' in sys.argv:
        fit_range()
    elif '--rejection' in sys.argv:
        rejection()
    elif '--population-drw' in sys.argv:
        population_drw_test()
    elif '--plate-replacement' in sys.argv:
        plate_replacement()
    else:
        main()
