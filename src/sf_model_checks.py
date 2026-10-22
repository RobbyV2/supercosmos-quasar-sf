import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, f as fdist, norm

import ensemble_structure_function as esf
import fit_structure_function as fsf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data'
LOGX = np.linspace(-6, 6, 481)
SLOPE_ALPHA = np.round(np.arange(-0.5, 2.901, 0.05), 8)
ALPHA = SLOPE_ALPHA[SLOPE_ALPHA < 2]
GROUPS = fsf.GROUPS


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


def psd_grid():
    g = cached_grid(esf.cache_path('model_checks/psd_integrals_beta2.npz'), SLOPE_ALPHA, LOGX[np.abs(LOGX) < 4.01],
                    lambda x, a: psd_sf2(x, a, 2.))
    return RegularGridInterpolator((ALPHA, g.grid[1]), g.values[:len(ALPHA)], bounds_error=True)


def mean_lag(f, band):
    return (f[f'lag_weights_{band}']*f[f'lag_nodes_{band}']).sum(-1)


def bin_lag(a, lag):
    return np.nanmean(np.where(a[...,0] > 0, lag, np.nan), axis=0)


def samples():
    """Fiducial sample, median-split groups and tercile grid cells, per band

    Yields
        band, sample name
        pair sums (quasar, bin, 3): pairs, dm^2, noise plus residual variance
        properties: LOGMBH, LOGLEDD_RATIO, z
        mean pair lag (quasar, bin), rest-frame days
    """
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
        if not np.array_equal(a[...,0],boot[f'lag_counts_{band}']):
            raise ValueError('sampling and ensemble archives disagree')
        p = prop.reindex(oid).copy()
        p['z'] = catalog[catalog.band.eq(band)].set_index('OBJID').reindex(oid).z
        lag = mean_lag(boot, band)
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
            yield band,name,a[keep],p.iloc[np.flatnonzero(keep)],lag[keep]


def adopted_reps(band, name, a):
    return fsf._object_amplitude(esf.object_bootstrap(a, *fsf.bootstrap_plan(band, name))[1])


def covariance(reps):
    sd = np.std(reps,axis=0,ddof=1)
    return esf.ledoit_wolf(reps/sd)[0]*np.outer(sd,sd)


def all_bin_fits(a, grid, reps, lag):
    """DRW and free-slope PSD fits to every bin with SF > 0, at each bin's mean pair lag

    a     pair sums (quasar, bin, slots), slots 0-2 as in accumulate
    reps  bootstrap SF replicates (n_boot, bin), mag
    lag   mean pair lag (quasar, bin), rest-frame days
    Returns
        fit rows, drw then bending_psd
        residual rows per model and bin
    """
    sf = np.sqrt(np.maximum(esf.object_second_moments(a).mean(axis=0),0))
    use = sf > 0
    dt,y,n = fsf.CENTERS[:len(sf)][use],sf[use],int(use.sum())
    lag = bin_lag(a, lag)[use]
    err,C = reps[:,use].std(axis=0,ddof=1),covariance(reps[:,use])
    w,full,long = err**-2,cho_factor(C),dt > fsf.LONG_MIN_D
    prec = np.sqrt(np.diag(cho_solve(full,np.eye(n))))
    lo,hi = grid.grid[1][[0,-1]]
    shape = lambda t,alpha:np.sqrt(-np.expm1(-lag/10**t) if alpha == 0 else
                                   grid(np.column_stack([np.full(n,alpha),np.clip(np.log10(lag)-t,lo,hi)])))
    def trial(t,alpha):
        u = shape(t,alpha)
        amp = (w*u)@y/((w*u)@u)
        return float(w@(y-amp*u)**2),amp
    def profile(alpha):
        ts = np.linspace(1,5,65)
        k = int(np.argmin([trial(t,alpha)[0] for t in ts]))
        best = minimize_scalar(lambda t:trial(t,alpha)[0],bounds=(ts[max(k-1,0)],ts[min(k+1,len(ts)-1)]),method='bounded')
        t = min([best.x,ts[0],ts[-1]],key=lambda q:trial(q,alpha)[0])
        cost,amp = trial(t,alpha)
        return dict(alpha=float(alpha),chi2=cost,logtau=t,amp=amp)
    profiles = [profile(x) for x in ALPHA]
    fits,resid = {},[]
    for name,prof in [('drw',[profiles[int(np.flatnonzero(ALPHA == 0)[0])]]),('bending_psd',profiles)]:
        p = min(prof,key=lambda q:q['chi2'])
        free,tau = len(prof) > 1,10**p['logtau']
        model = lambda q:q[0]*shape(q[1],p['alpha'])
        q = np.array([p['amp'],p['logtau']])
        J = np.column_stack([(model(q+d)-model(q-d))/(2*d.sum()) for d in np.diag([1e-6*q[0],1e-5])])
        A = np.linalg.solve(J.T@(w[:,None]*J),(w[:,None]*J).T)
        P = A@C@A.T
        e = np.sqrt(np.diag(P))
        accepted = [x['alpha'] for x in prof if x['chi2'] <= p['chi2']+1] if free else []
        k = 2+free
        pred = model(q)
        r = y-pred
        ci = cho_solve(full,r)
        x2 = r@ci
        row = fits[name] = dict(model=name,n_bins=n,n_params=k,dof=n-k,chi2_diag=p['chi2'],reduced_chi2_diag=p['chi2']/(n-k),
            chi2_full=x2,reduced_chi2_full=x2/(n-k),pvalue_full=chi2.sf(x2,n-k),sf_inf_mag=p['amp'],sf_inf_err_mag=e[0],
            tau_days=tau,tau_err_days=tau*np.log(10)*e[1],alpha=p['alpha']+0.,alpha_profile_lo=min(accepted,default=np.nan),
            alpha_profile_hi=max(accepted,default=np.nan),alpha_boundary=free and p['alpha'] in ALPHA[[0,-1]],
            alpha_profile_truncated=bool(accepted) and (min(accepted)==ALPHA[0] or max(accepted)==ALPHA[-1]),
            tau_boundary=tau < 10.01 or tau > 99990)
        if not free:
            R = np.eye(n)-J@A
            # parameter term scaled by max(reduced chi2, 1)
            cr = R@C@R.T+(max(x2/(n-k),1.)-1)*J@P@J.T
            x = float(r[long]@np.linalg.solve(cr[np.ix_(long,long)],r[long]))
            pl = chi2.sf(x,long.sum())
            row.update(final_lag_days=dt[-1],final_resid_mag=r[-1],long_n_bins=int(long.sum()),
                       final_resid_err_mag_param_scaled=np.sqrt(cr[-1,-1]),final_sigma_param_scaled=r[-1]/np.sqrt(cr[-1,-1]),
                       long_chi2_param_scaled=x,long_pvalue_param_scaled=pl,long_sigma_param_scaled=norm.isf(pl/2))
        resid += [dict(model=name,lag_center_days=t,lag_days=l,sf_mag=v,sf_err_mag=s,model_mag=m,relative_residual=v/m-1,
                       relative_residual_err=s/m,chi2_contribution=c2,conditional_pull=pull)
                  for t,l,v,s,m,c2,pull in zip(dt,lag,y,err,pred,r*ci,ci/prec)]
    B = fits['bending_psd']
    d = fits['drw']['chi2_diag']-B['chi2_diag']
    pv,pf = chi2.sf(d,1),fdist.sf(d/max(B['reduced_chi2_diag'],1.),1,B['dof'])
    B.update(vs_drw_delta_chi2=d,vs_drw_pvalue=pv,vs_drw_sigma=norm.isf(pv/2),vs_drw_f_pvalue=pf,vs_drw_f_sigma=norm.isf(pf/2))
    return list(fits.values()),resid


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


def greedy(a,stat,steps=2,key=lambda s:s):
    keep,path=np.ones(len(a),bool),[(-1,stat(a))]
    for _ in range(steps):
        trial={i:stat(a[keep&(np.arange(len(a))!=i)]) for i in np.flatnonzero(keep)}
        i=min(trial,key=lambda i:key(trial[i]))
        keep[i]=False
        path.append((i,trial[i]))
    return path


def lowmass_influence():
    grid,rows=psd_grid(),[]
    keys=['final_resid_mag','final_resid_err_mag_param_scaled','final_sigma_param_scaled','long_sigma_param_scaled']
    for band,name,a,prop,lag in samples():
        if name not in GROUPS[:2]:
            continue
        ids=prop.index.to_numpy(str)
        def drw(i):
            return next(f for f in all_bin_fits(a[i],grid,adopted_reps(band,name,a[i]),lag[i])[0] if f['model']=='drw')
        (_,f),*drops=greedy(np.arange(len(a)),drw,key=lambda f:f['long_sigma_param_scaled'])
        row=dict(band=band,sample=name,n_objects=len(a),final_lag_days=f['final_lag_days'],long_n_bins=f['long_n_bins'],**{k:f[k] for k in keys})
        for n,(i,d) in enumerate(drops,1):
            row.update({f'drop{n}_objid':ids[i],**{f'drop{n}_{k}':d[k] for k in keys}})
        rows.append(row)
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
            nt=esf.clean_nightly(load_completeness_lightcurves(ids)).query('band == @b')
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
        nt = esf.condense_nightly(s)
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
    with np.load(OUT/'structure_function_sampling_covariance.npz') as f:
        pairs = {b: f[f'fiducial_asymmetry_acc_{b}'][:, :2, 0, :, 0] for b in esf.BANDS}
    fid = {band: (a, adopted_reps(band, name, a), prop) for band, name, a, prop, _ in samples() if name == 'fiducial'}
    ref = pd.read_parquet(OUT/'plate_completeness_sample.parquet').astype({'OBJID': str}).set_index(['band', 'OBJID']).reference_mag
    mags = {b: ref[b].reindex(fid[b][2].index) for b in esf.BANDS}
    stars = star_floor({b: m.min() for b, m in mags.items()}, {b: m.max() for b, m in mags.items()})
    noise = pd.read_csv(OUT/'structure_function_noise_fraction.csv').set_index(['band', 'bin']).noise_over_sf2
    rows = []
    for b in esf.BANDS:
        a, r, prop = fid[b]
        nb, ids = a.shape[1], prop.index.to_numpy(str)
        nt = esf.clean_nightly(load_completeness_lightcurves(ids)).query('band == @b')
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
    fail = np.flatnonzero(~wide('shift_over_sigma').abs().lt(1).all(axis=1).to_numpy()[:end+1])
    fl.assign(criterion_start_days=C[fail[-1]+1 if len(fail) else 0], criterion_end_days=C[end]).to_csv(
        OUT/'structure_function_fit_range_floor.csv', index=False)


def fiducial_nightly():
    from plate_completeness import load_completeness_lightcurves
    selected = [x for x in samples() if x[1] == 'fiducial']
    lc = load_completeness_lightcurves(np.unique(np.concatenate([x[3].index for x in selected])))
    nightly = esf.clean_nightly(lc)
    for band, _, base, p, _ in selected:
        n = nightly[nightly.band.eq(band) & nightly.OBJID.isin(p.index)].reset_index(drop=True)
        run = lambda x, ids=p.index.to_numpy(): esf.accumulate(x, ids, p.z, bands=[band], edges=esf.EDGES[:base.shape[1]+1], resid=True)[:, 0]
        real = run(n)
        np.testing.assert_allclose(real[..., :3], base[..., :3], rtol=1e-10, atol=1e-8)
        yield band, base, p, n, run, real, lc


def replace_plate_epochs(n, rng, variant):
    """Plate magnitudes replaced by CCD-era values of the same quasar

    ccd_night   random CCD night
    ccd_survey  random CCD night, each survey equally likely
    ccd_median  CCD nightly median
    Gaussian noise brings each to the plate noise plus residual plate variance
    """
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


def plate_replacement(n_rep=200):
    fixed = pd.read_csv(OUT/'structure_function_fixed_population.csv').query('sample == "fixed"')
    with np.load(OUT/'ensemble_sf_covariance.npz') as f:
        adopted = {b: f[f'replicates_{b}'] for b in esf.BANDS}
    rows, variants = [], ['ccd_night', 'ccd_survey', 'ccd_median']
    for bi, (band, base, p, n, run, real, _) in enumerate(fiducial_nightly()):
        rng = np.random.default_rng(10050+bi)
        draws = {v: np.stack([run(replace_plate_epochs(n, rng, v)) for _ in range(n_rep)]) for v in variants}
        fixed_slots = [0, 2, 3, 4, 5]
        for d in draws.values():
            np.testing.assert_array_equal(d[..., fixed_slots], np.broadcast_to(real[..., fixed_slots], d[..., fixed_slots].shape))
        stack = np.stack([real, *(d.mean(axis=0) for d in draws.values())], axis=1)
        bounds, boot = (np.sqrt(np.maximum(x, 0)) for x in esf.object_bootstrap(stack, *fsf.bootstrap_plan(band, 'fiducial')))
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
    narrow,properties,narrow_curves,all_fits,all_resid=[],[],[],[],[]
    for index,(band,name,a,prop,lag) in enumerate(samples()):
        print(f'{band} {name}: {len(a)} objects',flush=True)
        ff,rr=all_bin_fits(a,grid,adopted_reps(band,name,a),lag)
        all_fits+=[{'band':band,'sample':name,'n_objects':len(a),**x} for x in ff]
        all_resid+=[{'band':band,'sample':name,**x} for x in rr]
        nr,pr,nc=nested_checks(a,prop,2928+index)
        narrow.extend([dict(band=band,sample=name,**p) for p in nr])
        properties.extend([dict(band=band,sample=name,**p) for p in pr])
        narrow_curves.extend([dict(band=band,sample=name,**p) for p in nc])
    pd.DataFrame(narrow).to_csv(OUT/'structure_function_narrowing_checks.csv',index=False)
    pd.DataFrame(properties).to_csv(OUT/'structure_function_narrowing_properties.csv',index=False)
    pd.DataFrame(narrow_curves).to_csv(OUT/'structure_function_narrowing_curves.csv',index=False)
    pd.DataFrame(all_fits).to_csv(OUT/'structure_function_all_bin_fits.csv',index=False)
    pd.DataFrame(all_resid).to_csv(OUT/'structure_function_all_bin_residuals.csv',index=False)

if __name__ == '__main__':
    esf.check_flags(__file__)
    if '--lowmass-influence' in sys.argv:
        lowmass_influence()
    elif '--band-ratio' in sys.argv:
        band_ratio()
    elif '--group-trends' in sys.argv:
        group_trends()
    elif '--noise-fraction' in sys.argv:
        noise_fraction()
    elif '--fit-range' in sys.argv:
        fit_range()
    elif '--plate-replacement' in sys.argv:
        plate_replacement()
    else:
        main()
