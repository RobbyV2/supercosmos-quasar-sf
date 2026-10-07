from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd

import assemble_lightcurves as al
import ensemble_structure_function as esf
import star_null_structure_function as sn


ROOT = Path(__file__).resolve().parents[1]
NATIVE_WORK = ROOT / "temp/native_sed_calibration"
NATIVE_BASE = NATIVE_WORK
NATIVE_VARIANT = "primary"
NATIVE_BAND = {"SERC-J/EJ": "bj", "SERC-R/AAO-R": "r", "POSSI-E(S)": "e"}
NATIVE_ALPHA = {"bj": 2, "e": 0, "r": 0, "tp": 0}


FIELD_NAMES=("fx","fy","frr","fx2_y2","fxy")
FIELD_COLUMNS=tuple(f"field_{k}" for k in range(17))


def native_field(xi,eta,spatial):
    if spatial in (11,17):
        from scipy.interpolate import BSpline
        step=2 if spatial==11 else 1
        xknots=np.r_[np.repeat(-4.,4),np.arange(-4.+step,4.,step),np.repeat(4.,4)]
        yknots=np.r_[np.repeat(-2.,4),np.arange(-2.+step,2.,step),np.repeat(2.,4)]
        x=BSpline.design_matrix(xi,xknots,3,extrapolate=True).toarray()[:,:-1]
        y=BSpline.design_matrix(eta,yknots,3,extrapolate=True).toarray()[:,:-1]
        return np.column_stack([x,y,xi*eta])
    return np.column_stack([xi,eta,xi*xi+eta*eta,xi*xi-eta*eta,xi*eta])[:,:spatial]


def native_design(m, xi, eta, center, scale, degree, spatial):
    return np.column_stack([np.polynomial.legendre.legvander((m-center)/scale, degree),
                            native_field(xi, eta, spatial)])


def native_robust_fit(A, y):
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(8):
        r = y-A@c
        scale = max(al._mad_sigma(r), .01)
        w = np.sqrt(np.minimum(1., 1.345*scale/np.maximum(np.abs(r), 1e-12)))
        c = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)[0]
    return c, scale


def native_inverse(raw, xi, eta, p):
    c = np.array([p[f"c{k}"] for k in range(4)])
    count=int(p.spatial)
    names=FIELD_COLUMNS if FIELD_COLUMNS[0] in p else FIELD_NAMES
    spatial = native_field(xi, eta, count) @ np.array([p[k] for k in names[:count]])
    def forward(m):
        return m+np.polynomial.legendre.legval((m-p.mag_center)/p.mag_scale, c)+spatial
    lo, hi = np.full(len(raw), p.mag_lo), np.full(len(raw), p.mag_hi)
    supported = (raw >= forward(lo)) & (raw <= forward(hi))
    supported &= (xi >= p.xi_lo) & (xi <= p.xi_hi) & (eta >= p.eta_lo) & (eta <= p.eta_hi)
    for _ in range(45):
        mid = (lo+hi)/2
        faint = forward(mid) < raw
        lo, hi = np.where(faint, mid, lo), np.where(faint, hi, mid)
    m = (lo+hi)/2
    derivative = 1+np.polynomial.legendre.legval((m-p.mag_center)/p.mag_scale,
                                               np.polynomial.legendre.legder(c))/p.mag_scale
    return np.where(supported, m, np.nan), derivative, supported


def native_inputs():
    stars = pd.read_parquet(ROOT / "data/standard_star_sed.parquet").set_index("star")
    stars=stars.join(pd.read_parquet(ROOT/"data/standard_star_sed_response.parquet").set_index("star"),validate="one_to_one")
    if NATIVE_VARIANT != "primary":
        for band in ("bj", "r", "e", "tp"):
            variant=NATIVE_VARIANT
            if variant=="effective":
                variant={-1:"energy",0:"primary",1:"tilt1",2:"tilt2"}[NATIVE_ALPHA[band]]
            column = f"{band}_native_{variant}_shift"
            if column in stars:
                stars[band+"_native"] += stars[column]
        if NATIVE_VARIANT=="effective":
            stars["r_native"]+=stars.r_native_og590_shift
            stars["r_native_err"]=stars.r_native_og590_err
    if not stars.index.is_unique:
        raise ValueError("Duplicate stellar SED identity")
    det = al.drop_blended_parents(al.star_detections_cached().rename(columns=al._STAR_BLEND)).rename(
        columns={v:k for k,v in al._STAR_BLEND.items()})
    det = det[det.SURVEYNAME.isin(NATIVE_BAND)].copy()
    det["native"] = det.SURVEYNAME.map(NATIVE_BAND)
    det.loc[det.plate.isin((131900,131903)),"native"] = "tp"
    det["band"] = det.SURVEYNAME.map(sn.SURVEY_SDSS_BAND)
    native = np.where(det.possi, "R1", det.bi.map(al._SSS_BI_BAND))
    det = det[al.saturation_mask(det.SMAG.to_numpy(), native)].copy()
    x = stars.reindex(det.star)
    det["fit_ok"] = x.fit_ok.fillna(False).to_numpy(bool)
    det["reference"] = np.where(det.band.eq("g"), x.g, x.r)
    det["color"] = (x.g-x.r).to_numpy()
    for a,b in zip("ugri","griz"):
        det[a+"_"+b] = (x[a]-x[b]).to_numpy()
    for key, suffix in (("expected", "_native"), ("expected_error", "_native_err")):
        det[key] = np.select([det.native.eq(b) for b in ("bj","r","e","tp")],
                             [x[b+suffix].to_numpy() for b in ("bj","r","e","tp")],default=np.nan)
    det["fold"] = (det.star.to_numpy(np.int64)*2654435761) % 5
    det["xi"], det["eta"] = al.field_coords(det.plate.to_numpy(), det.ra.to_numpy(), det.dec.to_numpy())
    return stars, det.reset_index(drop=True)


def native_calibration():
    NATIVE_WORK.mkdir(exist_ok=True, parents=True)
    stars, det = native_inputs()
    coefficients, validation, held, coverage = [], [], [], []
    folds = (stars.index.to_numpy(np.int64)*2654435761) % 5
    eligible = stars.fit_ok.to_numpy(bool) & (folds != 4)
    centers = al.plate_centres()
    for plate, d in det.groupby("plate", sort=True):
        d = d[d.fit_ok & np.isfinite(d.expected+d.expected_error)].copy()
        train = d.fold.ne(4)
        if train.sum() < 1000:
            coverage.append(dict(plate=plate, status="fewer than 1000 fitting detections"))
            continue
        box = {name+"_"+side: float(np.quantile(d.loc[train,name], q))
               for name in ("xi","eta") for side,q in (("lo",0.),("hi",1.))}
        xi, eta = al.field_coords(np.full(len(stars),plate), stars.ra.to_numpy(), stars.dec.to_numpy(), centers)
        inside = eligible & (xi >= box["xi_lo"]) & (xi <= box["xi_hi"]) & (eta >= box["eta_lo"]) & (eta <= box["eta_hi"])
        column = d.native.iloc[0]+"_native"
        expected = stars[column].to_numpy()
        edges = np.arange(13., 23.26, .25)
        mid = (edges[:-1]+edges[1:])/2
        total = np.histogram(expected[inside], edges)[0]
        seen = np.histogram(d.loc[train & d.star.isin(stars.index[inside]), "expected"], edges)[0]
        rate = np.divide(seen,total,out=np.zeros(len(total),float),where=total>0)
        reliable = total >= 100
        peak = np.max(rate[reliable]) if reliable.any() else 0
        complete = reliable & (rate >= .95*min(peak,1.))
        groups = np.split(np.flatnonzero(complete), np.flatnonzero(np.diff(np.flatnonzero(complete))>1)+1)
        groups = [g for g in groups if len(g)>=6]
        if not groups or peak < .85:
            coverage.append(dict(plate=plate, status="no sufficiently complete native interval", detection_peak=peak))
            continue
        fit = max(groups, key=lambda g:total[g].sum())
        lo,hi = edges[fit[0]], edges[fit[-1]+1]
        center, scale = (lo+hi)/2,(hi-lo)/2
        support = d.expected.between(lo,hi) & d.xi.between(box["xi_lo"],box["xi_hi"]) & d.eta.between(box["eta_lo"],box["eta_hi"])
        learn = d[train & support]
        y = (learn.SMAG-learn.expected).to_numpy()
        trials = []
        grid = np.linspace(-1,1,200)
        for degree in (1,2,3):
            for spatial in (0,2,3,5,11,17):
                A = native_design(learn.expected.to_numpy(), learn.xi.to_numpy(), learn.eta.to_numpy(), center,scale,degree,spatial)
                losses, valid = [], True
                for fold in range(4):
                    pick = learn.fold.to_numpy()!=fold
                    c, sig = native_robust_fit(A[pick],y[pick])
                    derivative = 1+np.polynomial.legendre.legval(grid,np.polynomial.legendre.legder(c[:degree+1]))/scale
                    valid &= bool((derivative>.2).all())
                    r = (y[~pick]-A[~pick]@c)/sig
                    losses.append(float(np.mean(np.where(np.abs(r)<=1.345,.5*r*r,1.345*(np.abs(r)-.5*1.345)))*sig**2))
                trial = dict(plate=int(plate),degree=degree,spatial=spatial,loss=np.mean(losses),
                             loss_se=np.std(losses,ddof=1)/2,monotone=valid,
                             **{f"fold_loss_{k}":value for k,value in enumerate(losses)})
                trials.append(trial)
        permitted = [t for t in trials if t["monotone"]]
        if not permitted:
            coverage.append(dict(plate=plate,status="no monotone response"))
            continue
        best = min(permitted,key=lambda t:t["loss"])
        paired_se=lambda t:np.std([t[f"fold_loss_{k}"]-best[f"fold_loss_{k}"] for k in range(4)],ddof=1)/2
        selected = min((t for t in permitted if t["loss"]-best["loss"]<=paired_se(t)+1e-15),
                       key=lambda t:(t["degree"]+t["spatial"],t["loss"]))
        degree,spatial = selected["degree"],selected["spatial"]
        A = native_design(learn.expected.to_numpy(),learn.xi.to_numpy(),learn.eta.to_numpy(),center,scale,degree,spatial)
        c,sig = native_robust_fit(A,y)
        p = pd.Series(dict(plate=int(plate),survey=d.SURVEYNAME.iloc[0],native=d.native.iloc[0],band=d.band.iloc[0],
                           degree=degree,spatial=spatial,mag_lo=lo,mag_hi=hi,mag_center=center,mag_scale=scale,
                           n_train=len(learn),fit_scatter=sig,**box,
                           **dict(zip([f"c{k}" for k in range(4)],np.pad(c[:degree+1],(0,3-degree)))),
                           **dict(zip(FIELD_COLUMNS,np.pad(c[degree+1:],(0,17-spatial))))))
        if (1+np.polynomial.legendre.legval(grid,np.polynomial.legendre.legder(p[["c0","c1","c2","c3"]].to_numpy(float)))/scale<=.2).any():
            coverage.append(dict(plate=plate,status="final response nonmonotone"))
            continue
        coefficients.append(p.to_dict())
        for t in trials:
            t["selected"] = t is selected
        validation.extend(trials)
        test = d[d.fold.eq(4)].copy()
        corrected,derivative,ok = native_inverse(test.SMAG.to_numpy(),test.xi.to_numpy(),test.eta.to_numpy(),p)
        test["corrected_native"],test["response_derivative"],test["supported"] = corrected,derivative,ok
        test["inverse_residual_mag"] = corrected-test.expected
        test["expected_supported"] = test.expected.between(lo,hi)&test.xi.between(box["xi_lo"],box["xi_hi"])&test.eta.between(box["eta_lo"],box["eta_hi"])
        A=native_design(test.expected.to_numpy(),test.xi.to_numpy(),test.eta.to_numpy(),center,scale,degree,spatial)
        derivative_at_expected=1+np.polynomial.legendre.legval((test.expected-center)/scale,np.polynomial.legendre.legder(c[:degree+1]))/scale
        test["forward_residual_raw_mag"] = test.SMAG-test.expected-A@c
        test["residual_mag"] = test.forward_residual_raw_mag/derivative_at_expected
        held.append(test)
        coverage.append(dict(plate=int(plate),status="fitted",n_train=len(learn),n_test=len(test),n_test_supported=int(ok.sum()),
                             mag_lo=lo,mag_hi=hi,detection_peak=peak))
        print(f"{plate} {p['survey']}: {len(learn)} train, {ok.sum()}/{len(test)} supported test, degree {degree}, spatial {spatial}, native {lo:.2f}-{hi:.2f}",flush=True)
    pd.DataFrame(coefficients).to_csv(NATIVE_WORK/"coefficients.csv",index=False)
    pd.DataFrame(validation).to_csv(NATIVE_WORK/"model_selection.csv",index=False)
    pd.DataFrame(coverage).to_csv(NATIVE_WORK/"coverage.csv",index=False)
    if held:
        pd.concat(held,ignore_index=True).to_parquet(NATIVE_WORK/"heldout.parquet",index=False)
    inputs = [ROOT/"src/plate_magnitude_models.py", ROOT/"data/standard_star_sed.parquet",
              ROOT/"data/standard_star_sed_response.parquet"]
    inputs.extend(ROOT/"data"/p for p in ("serc-j.txt","serc-r.txt","possi-e.txt","techpan_og590.txt",
        "iiiaf_og590.txt","SLOAN_SDSS.g.dat","SLOAN_SDSS.r.dat","alpha_lyr_stis_012.fits"))
    (NATIVE_WORK/"provenance.json").write_text(json.dumps(dict(variant=NATIVE_VARIANT,
        sha256={str(p.relative_to(ROOT)):hashlib.file_digest(p.open("rb"),"sha256").hexdigest() for p in inputs}),indent=2)+"\n")


def native_quasars():
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    from dustmaps.sfd import SFDQuery
    from stellar_sed_calibration import redden
    cache = NATIVE_BASE/"raw_quasars.parquet"
    if cache.exists():
        q = pd.read_parquet(cache)
    else:
        raw = al.assign_group_ids_by_sky(al.sss_raw())
        mapping = al.match_groups_to_catalog(raw, str(ROOT/"data/S82/Catalog.parquet"))
        q = al.drop_blended_parents(raw.drop(columns=["OBJID","z"],errors="ignore").merge(mapping,on="GROUP_ID",how="inner"))
        q = q[q.SURVEYNAME.isin(NATIVE_BAND)].copy()
        q = q[al.saturation_mask(q.SMAG.to_numpy(),np.where(q.SURVEYNAME.eq("POSSI-E(S)"),"R1",q.BAND_NATIVE))]
        q.to_parquet(cache,index=False)
    q["OBJID"] = q.OBJID.astype(str)
    q["native"] = q.SURVEYNAME.map(NATIVE_BAND)
    q.loc[q.PLATEID.isin((131900,131903)),"native"] = "tp"
    q["band"] = q.SURVEYNAME.map(sn.SURVEY_SDSS_BAND)
    coefficients = pd.read_csv(NATIVE_WORK/"coefficients.csv").set_index("plate")
    xi,eta = al.field_coords(q.PLATEID.to_numpy(),q.RA.to_numpy(),q.DEC.to_numpy())
    q["native_mag"],q["response_derivative"],q["supported"] = np.nan,np.nan,False
    for plate, idx in q.groupby("PLATEID").groups.items():
        if plate not in coefficients.index:
            continue
        loc=q.index.get_indexer(idx)
        result=native_inverse(q.loc[idx,"SMAG"].to_numpy(),xi[loc],eta[loc],coefficients.loc[plate])
        for name,values in zip(("native_mag","response_derivative","supported"),result):
            q.loc[idx,name]=values
    dust_path=NATIVE_BASE/"quasar_dust.csv"
    if not dust_path.exists():
        cat=pd.read_parquet(ROOT/"data/S82/Catalog.parquet")
        ebv=SFDQuery(map_dir=str(ROOT/"data/sfd"))(SkyCoord(cat.RA,cat.DEC,unit="deg"))
        pd.DataFrame(dict(OBJID=cat.objectId.astype(str),ra=cat.RA,dec=cat.DEC,
                          ebv_sfd=ebv,ebv=.86*ebv,previous_catalog_ebv=cat.ebv)).to_csv(dust_path,index=False)
    dust=pd.read_csv(dust_path).set_index("OBJID")
    dust.index=dust.index.astype(str)
    objects=q[["OBJID","z"]].drop_duplicates("OBJID").set_index("OBJID")
    objects["ebv"] = dust.ebv.reindex(objects.index)
    if objects.ebv.isna().any():
        raise ValueError("Missing consistently scaled quasar foreground reddening")
    template=Table.read(ROOT/"data/vandenberk_qso_composite_cds.txt",format="ascii.cds")
    wave,flux=np.asarray(template["Wave"],float),np.asarray(template["FluxD"],float)
    bands={b:al.read_bandpass(name) for b,name in {"bj":"serc-j.txt","r":"serc-r.txt","e":"possi-e.txt","tp":"techpan_og590.txt","sdss_g":"SLOAN_SDSS.g.dat","sdss_r":"SLOAN_SDSS.r.dat"}.items()}
    if NATIVE_VARIANT == "energy":
        for b in ("bj","r","e"):
            w,t=bands[b]
            bands[b]=(w,t/w)
    if NATIVE_VARIANT == "effective":
        bands["r"]=al.read_bandpass("iiiaf_og590.txt")
        for b in ("bj","r","e","tp"):
            w,t=bands[b]
            bands[b]=(w,t*(w/np.average(w,weights=t))**NATIVE_ALPHA[b])
    vega={b:al.ab_minus_vega(*bands[b]) for b in ("bj","r","e","tp")}
    terms=[]
    for obj,p in objects.iterrows():
        w=wave*(1+p.z);use=(w>=2000)&(w<=33000)
        w=w[use]
        f=redden(w,flux[use]/(1+p.z),p.ebv)
        w=al.vac_to_air(w)
        mags={b:al.abmag(w,f,*bp) for b,bp in bands.items()}
        terms.append(dict(OBJID=obj,ebv=p.ebv,
                          bj=mags["sdss_g"]-mags["bj"]+vega["bj"],
                          r=mags["sdss_r"]-mags["r"]+vega["r"]-.015,
                          e=mags["sdss_r"]-mags["e"]+vega["e"]-.015,
                          tp=mags["sdss_r"]-mags["tp"]+vega["tp"]-.015))
    terms=pd.DataFrame(terms).set_index("OBJID")
    terms.to_csv(NATIVE_WORK/"quasar_sed_terms.csv")
    matched=terms.reindex(q.OBJID)
    q["sed_term"] = np.select([q.native.eq(b) for b in ("bj","r","e","tp")],[matched[b].to_numpy() for b in ("bj","r","e","tp")],default=np.nan)
    pogson=q.native_mag+q.sed_term
    softening=np.where(q.band.eq("g"),9e-11,1.2e-10)
    q["mag"] = -2.5/np.log(10)*(np.arcsinh(10**(-.4*pogson)/(2*softening))+np.log(softening))
    q["supported"] = q.supported.astype(bool)&np.isfinite(q.mag)
    q.to_parquet(NATIVE_WORK/"quasar_epochs.parquet",index=False)
    print(q.groupby(["band","supported"]).size().to_string())


def native_validation():
    held = pd.read_parquet(NATIVE_WORK/"heldout.parquet")
    old = NATIVE_BASE/"baseline/stars.parquet"
    columns = ["star","plate","old_residual_mag","SMAG_ERR","reference_variance"]
    d = (held.merge(pd.read_parquet(old)[columns],on=["star","plate"],how="left",validate="one_to_one") if old.exists()
         else held.assign(**dict.fromkeys(columns[2:],np.nan)))
    d["old_individual_sigma"] = np.sqrt(d.SMAG_ERR**2+d.reference_variance)
    d["new_individual_sigma"] = d.old_individual_sigma
    d["in_fiducial_range"] = False
    limits = sn._fiducial_limits()
    for band in "gr":
        d.loc[d.band.eq(band),"in_fiducial_range"] = d.loc[d.band.eq(band),"reference"].between(*limits.loc[band])
    bins=[]
    for (survey,plate),g in d[d.expected_supported].groupby(["SURVEYNAME","plate"]):
        for axis,width in (("reference",.25),("u_g",.25),("g_r",.25),("r_i",.25),("i_z",.25),("xi",.5),("eta",.25)):
            for cell,q in g[g.in_fiducial_range].groupby(np.floor(g.loc[g.in_fiducial_range,axis]/width)):
                for model,column in (("old","old_residual_mag"),("native","residual_mag")):
                    value=q[column].dropna()
                    if len(value)<30:continue
                    bins.append(dict(survey=survey,plate=plate,axis=axis,lo=cell*width,hi=(cell+1)*width,
                                     model=model,n=len(value),median=value.median(),mean=value.mean(),nmad=al._mad_sigma(value),
                                     median_error=1.2533*al._mad_sigma(value)/np.sqrt(len(value)),
                                     individual_sigma=q["new_individual_sigma" if model=="native" else "old_individual_sigma"].median()))
    bins=pd.DataFrame(bins)
    bins.to_csv(NATIVE_WORK/"validation_bins.csv",index=False)
    rows=[]
    for survey,g in d.groupby("SURVEYNAME"):
        accepted=g.expected_supported&g.in_fiducial_range&np.isfinite(g.old_residual_mag)
        for model,column in (("old","old_residual_mag"),("native","residual_mag")):
            values=g.loc[accepted,column]
            z=bins[bins.survey.eq(survey)&bins.model.eq(model)&bins.axis.eq("reference")]
            rows.append(dict(survey=survey,model=model,n_common=len(values),n_all_held=len(g),n_supported=int(g.supported.sum()),
                             median=values.median(),mean=values.mean(),nmad=al._mad_sigma(values),
                             plate_bin_rms=np.sqrt(np.mean(z["median"]**2)),plate_bin_max=z["median"].abs().max(),
                             bin_medians_above_individual_sigma=int((z["median"].abs()>z.individual_sigma).sum())))
    summary=pd.DataFrame(rows)
    summary.to_csv(NATIVE_WORK/"validation_summary.csv",index=False)
    d.to_parquet(NATIVE_WORK/"heldout_validation.parquet",index=False)
    if NATIVE_VARIANT=="effective":
        for name in ("bins","summary"):shutil.copy2(NATIVE_WORK/f"validation_{name}.csv",ROOT/f"data/plate_native_validation_{name}.csv")
    print(summary.to_string(index=False))


def native_noise():
    from scipy.optimize import nnls
    stars,det=native_inputs()
    coefficients=pd.read_csv(NATIVE_WORK/"coefficients.csv").set_index("plate")
    rows=[]
    for plate,d in det[det.fit_ok].groupby("plate"):
        if plate not in coefficients.index:continue
        p=coefficients.loc[plate]
        use=d.expected.between(p.mag_lo,p.mag_hi)&d.xi.between(p.xi_lo,p.xi_hi)&d.eta.between(p.eta_lo,p.eta_hi)
        d=d[use].copy()
        m,derivative,supported=native_inverse(d.SMAG.to_numpy(),d.xi.to_numpy(),d.eta.to_numpy(),p)
        c=p[["c0","c1","c2","c3"]].to_numpy(float)
        at=(d.expected.to_numpy()-p.mag_center)/p.mag_scale
        response=1+np.polynomial.legendre.legval(at,np.polynomial.legendre.legder(c))/p.mag_scale
        count=int(p.spatial)
        names=FIELD_COLUMNS if FIELD_COLUMNS[0] in p else FIELD_NAMES
        field=native_field(d.xi.to_numpy(),d.eta.to_numpy(),count)@p[list(names[:count])].to_numpy(float)
        forward=(d.SMAG-d.expected-np.polynomial.legendre.legval(at,c)-field)/response
        d["residual"]=np.where(supported,m-d.expected,forward)
        d["inverse_supported"]=supported
        d["boundary_distance"]=np.minimum(d.expected-p.mag_lo,p.mag_hi-d.expected)
        d["boundary_sigma"]=d.boundary_distance/(p.fit_scatter/response)
        d["reference_variance"]=np.where(d.band.eq("g"),stars.g_err.reindex(d.star),stars.r_err.reindex(d.star))**2
        rows.append(d)
    epochs=pd.concat(rows,ignore_index=True)
    epochs.to_parquet(NATIVE_WORK/"star_epochs.parquet",index=False)
    e=epochs.sort_values(["star","band"],kind="stable").reset_index(drop=True)
    i,j=esf.group_pairs(e.star.to_numpy()*2+e.band.eq("r").to_numpy())
    code=e.native.map({"bj":0,"r":1,"e":2,"tp":3}).to_numpy()*100+np.floor(e.expected.to_numpy()*2).astype(int)
    pairs=pd.DataFrame(dict(star=e.star.to_numpy()[i],fold=e.fold.to_numpy()[i],
        a=np.minimum(code[i],code[j]),b=np.maximum(code[i],code[j]),
        mag=e.reference.to_numpy()[i],dm=e.residual.to_numpy()[i]-e.residual.to_numpy()[j],
        boundary=np.minimum(e.boundary_sigma.to_numpy()[i],e.boundary_sigma.to_numpy()[j])))
    pairs["bin"]=np.floor(pairs.mag*2).astype(int)
    noise,checks=[],[]
    training=pairs[pairs.fold.ne(4)&pairs.boundary.ge(3)]
    edges,y,weight,clip=[],[],[],{}
    for (a,b),q in training.groupby(["a","b"]):
        if q.star.nunique()<50:continue
        center=q.dm.median();limit=5*al._mad_sigma(q.dm)
        z=q.loc[(q.dm-center).abs()<=limit,"dm"]
        edges.append((a,b));y.append(np.mean(z**2));weight.append(np.sqrt(len(z)));clip[(a,b)]=(center,limit)
    active=np.unique(edges);lookup={k:i for i,k in enumerate(active)}
    A=np.zeros((len(edges),len(active)))
    for i,(a,b) in enumerate(edges):A[i,lookup[a]]+=1;A[i,lookup[b]]+=1
    if np.linalg.matrix_rank(A)<len(active):raise ValueError("Native error bins are not identifiable")
    variance=nnls(A*np.array(weight)[:,None],np.array(y)*weight)[0]
    v=dict(zip(active,variance))
    for k,value in v.items():
        native,mb=divmod(k,100)
        support=training[training.a.eq(k)|training.b.eq(k)]
        noise.append(dict(native=("bj","r","e","tp")[native],mag_lo=mb/2,mag_hi=(mb+1)/2,variance=value,sigma=np.sqrt(value),
                          n_training_pairs=len(support),n_training_stars=support.star.nunique()))
    for (a,b),q in pairs.groupby(["a","b"]):
        if a not in v or b not in v or (a,b) not in clip:continue
        center,limit=clip[(a,b)]
        for subset,sel in (("training",q.fold.ne(4)),("heldout",q.fold.eq(4))):
            for margin in (0,3):
                z=q[sel&q.boundary.ge(margin)]
                if len(z)<30:continue
                core=z.dm[(z.dm-center).abs()<=limit]
                checks.append(dict(native_a=("bj","r","e","tp")[a//100],native_b=("bj","r","e","tp")[b//100],
                    native_a_mag_lo=(a%100)/2,native_b_mag_lo=(b%100)/2,subset=subset,boundary_sigma_min=margin,n=len(z),n_core=len(core),
                    predicted_pair_variance=v[a]+v[b],pair_mean_square=np.mean(z.dm**2),core_mean_square=np.mean(core**2),
                    mean=z.dm.mean(),median=z.dm.median()))
    noise=pd.DataFrame(noise);noise.to_csv(NATIVE_WORK/"random_errors.csv",index=False)
    pd.DataFrame(checks).to_csv(NATIVE_WORK/"random_error_validation.csv",index=False)
    e["s2"]=np.nan
    for native,g in noise.groupby("native"):
        use=e.native.eq(native)
        at=e.loc[use,"expected"]+e.loc[use,"residual"]
        e.loc[use,"s2"]=np.interp(at,(g.mag_lo+g.mag_hi)/2,g.variance)
        e.loc[use&~e.expected.between(g.mag_lo.min(),g.mag_hi.max()),"s2"]=np.nan
    e=e[np.isfinite(e.s2)].copy()
    ccd=e.drop_duplicates(["star","band"]).copy()
    ccd["native"],ccd["residual"],ccd["s2"],ccd["boundary_sigma"]="ccd",0.,ccd.reference_variance,np.inf
    e=pd.concat([e,ccd],ignore_index=True).sort_values(["star","band"],kind="stable").reset_index(drop=True)
    i,j=esf.group_pairs(e.star.to_numpy()*2+e.band.eq("r").to_numpy())
    code=e.native.map({"ccd":0,"bj":1,"r":2,"e":3,"tp":4}).to_numpy()
    pairs=pd.DataFrame(dict(star=e.star.to_numpy()[i],fold=e.fold.to_numpy()[i],band=e.band.to_numpy()[i],
        a=np.minimum(code[i],code[j]),b=np.maximum(code[i],code[j]),mag=e.reference.to_numpy()[i],
        dm=e.residual.to_numpy()[i]-e.residual.to_numpy()[j],se2=e.s2.to_numpy()[i]+e.s2.to_numpy()[j],
        boundary=np.minimum(e.boundary_sigma.to_numpy()[i],e.boundary_sigma.to_numpy()[j])))
    pairs["bin"]=np.floor(pairs.mag*2).astype(int)
    floor,bootstrap=[],[]
    for (a,b,mb),g in pairs.groupby(["a","b","bin"]):
        training=g[g.fold.ne(4)&g.boundary.ge(3)]
        if len(training)<100:continue
        center=training.dm.median();limit=5*al._mad_sigma(training.dm)
        training=training[(training.dm-center).abs()<=limit]
        training_stars=(training.dm**2-training.se2).groupby(training.star).mean()
        variance=training_stars.mean()
        training_se=training_stars.std(ddof=1)/np.sqrt(len(training_stars))
        class_id=len(bootstrap)
        bootstrap.append(training_stars.rename(class_id))
        for subset,sel in (("training",g.fold.ne(4)),("heldout",g.fold.eq(4))):
            for margin in (0,3):
                q=g[sel&g.boundary.ge(margin)]
                if len(q)<30:continue
                core=q[(q.dm-center).abs()<=limit]
                bystar=(core.dm**2-core.se2-variance).groupby(core.star).mean()
                floor.append(dict(native_a=("ccd","bj","r","e","tp")[a],native_b=("ccd","bj","r","e","tp")[b],band=q.band.iloc[0],
                    mag_lo=mb/2,mag_hi=(mb+1)/2,subset=subset,boundary_sigma_min=margin,n_pairs=len(q),n_core=len(core),n_stars=len(bystar),
                    class_id=class_id,clip_center=center,clip_limit=limit,
                    residual_variance=variance,null_mean=np.mean(core.dm**2-core.se2-variance),unclipped_null_mean=np.mean(q.dm**2-q.se2-variance),
                    null_star_mean=bystar.mean(),training_floor_se=training_se,
                    null_star_se=np.hypot(bystar.std(ddof=1)/np.sqrt(len(bystar)),training_se)))
    pd.DataFrame(floor).to_csv(NATIVE_WORK/"pair_variance_validation.csv",index=False)
    pairs.to_parquet(NATIVE_WORK/"star_pairs.parquet",index=False)
    from scipy.sparse import coo_matrix
    ids=np.unique(np.concatenate([x.index.to_numpy() for x in bootstrap]));si={star:i for i,star in enumerate(ids)}
    ii=np.concatenate([[si[x] for x in value.index] for value in bootstrap])
    jj=np.repeat(np.arange(len(bootstrap)),[len(x) for x in bootstrap]);value=np.concatenate([x.to_numpy() for x in bootstrap])
    V=coo_matrix((value,(ii,jj)),shape=(len(ids),len(bootstrap))).tocsr()
    W=coo_matrix((np.ones(len(value)),(ii,jj)),shape=V.shape).tocsr()
    rng=np.random.default_rng(8141);draws=[]
    for _ in range(20):
        weights=rng.poisson(1,(20,len(ids)))
        draws.append((weights@V)/(weights@W))
    np.savez_compressed(NATIVE_WORK/"pair_variance_bootstrap.npz",draws=np.vstack(draws))
    print(noise.to_string(index=False))


def native_publish():
    if NATIVE_VARIANT!="effective":raise ValueError("Only the validated effective responses are adopted")
    mapping={"coefficients.csv":"plate_native_calibration.csv","random_errors.csv":"plate_native_random_errors.csv",
        "pair_variance_bootstrap.npz":"plate_native_pair_variance_bootstrap.npz",
        "random_error_validation.csv":"plate_native_random_error_validation.csv",
        "pair_variance_validation.csv":"plate_native_pair_variance_validation.csv","coverage.csv":"plate_native_coverage.csv"}
    for source,target in mapping.items():shutil.copy2(NATIVE_WORK/source,ROOT/"data"/target)
    floor=pd.read_csv(NATIVE_WORK/"pair_variance_validation.csv")
    floor=floor[floor.subset.eq("training")&floor.boundary_sigma_min.eq(3)].drop_duplicates("class_id")
    floor.to_csv(ROOT/"data/plate_native_pair_variance.csv",index=False)
    q=pd.read_parquet(NATIVE_WORK/"quasar_epochs.parquet")
    q["calibration_supported"]=q.supported
    q["error_supported"],q["magerr"]=False,np.nan
    noise=pd.read_csv(NATIVE_WORK/"random_errors.csv")
    for native,g in noise.groupby("native"):
        use=q.native.eq(native)
        q.loc[use,"magerr"]=np.sqrt(np.interp(q.loc[use,"native_mag"],(g.mag_lo+g.mag_hi)/2,g.variance))
        q.loc[use,"error_supported"]=q.loc[use,"native_mag"].between(g.mag_lo.min(),g.mag_hi.max())
    q["supported"]=q.calibration_supported&q.error_supported&np.isfinite(q.mag+q.magerr)&q.magerr.gt(0)
    q["support_reason"]=np.where(q.supported,"supported",np.where(q.calibration_supported,"random-error range","instrumental response or stellar footprint"))
    q.to_parquet(ROOT/"data/plate_native_quasar_epochs.parquet",index=False)
    inputs=[ROOT/"src/plate_magnitude_models.py",ROOT/"src/stellar_sed_calibration.py",
        ROOT/"data/roe_stars.hdf5",ROOT/"data/wu_qso.hdf5",ROOT/"data/survey_table.fits",
        ROOT/"data/standard_star_sed.parquet",ROOT/"data/standard_star_sed_response.parquet",
        ROOT/"data/vandenberk_qso_composite_cds.txt",
        ROOT/"data/sfd/SFD_dust_4096_ngp.fits",ROOT/"data/sfd/SFD_dust_4096_sgp.fits"]
    inputs.extend(ROOT/"data"/p for p in ("serc-j.txt","possi-e.txt","techpan_og590.txt","iiiaf_og590.txt",
        "SLOAN_SDSS.g.dat","SLOAN_SDSS.r.dat","alpha_lyr_stis_012.fits"))
    products=[ROOT/"data"/p for p in mapping.values()]+[ROOT/"data/plate_native_quasar_epochs.parquet",ROOT/"data/plate_native_pair_variance.csv"]
    manifest=dict(method="stellar SED native response; individual epoch inverse; quasar SED to SDSS",
        calibration_fold="star * 2654435761 modulo 5; fold 4 excluded from fits and CV",
        foreground="F99 R_V=3.1; 0.86 times original SFD",
        support="training stellar footprint and complete native interval; measured native error bins",
        sha256={str(p.relative_to(ROOT)):hashlib.file_digest(p.open("rb"),"sha256").hexdigest() for p in inputs+products},
        validation_scope="Fold 4 is excluded from coefficient fitting and training cross-validation; inspected during method development. Old calibration comparison is conditional on earlier all-star fits.",
        uncertainty_scope="Stellar residual-floor draws share star weights across classes. Quasar intervals condition on fitted instrumental response and random-error curves; template and lag-null tests are separate sensitivities.")
    (ROOT/"data/plate_native_calibration.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(q.groupby(["band","supported"]).size().to_string())


def native_longform(radius=1.0,sat_cut=True):
    if radius!=1.0 or not sat_cut:raise ValueError("Native calibration is validated for the adopted matching and saturation cuts")
    q=pd.read_parquet(ROOT/"data/plate_native_quasar_epochs.parquet")
    q=q[q.supported]
    return pd.DataFrame(dict(OBJID=q.OBJID.astype(str),band=q.band,time=q.MJD,mag=q.mag,magerr=q.magerr,
        survey=np.where(q.SURVEYNAME.eq("POSSI-E(S)"),"sss_possi","sss"),plate=q.PLATEID,
        smag=q.SMAG,calib_ok=True,trunc_v=1.))


def native_null():
    from scipy.sparse import coo_matrix
    e=pd.read_parquet(NATIVE_WORK/"star_epochs.parquet")
    e=e[e.fold.eq(4)&e.boundary_sigma.ge(3)].copy()
    e["s2"]=np.nan
    noise=pd.read_csv(ROOT/"data/plate_native_random_errors.csv")
    for native,g in noise.groupby("native"):
        use=e.native.eq(native)&e.expected.between(g.mag_lo.min(),g.mag_hi.max())
        e.loc[use,"s2"]=np.interp(e.loc[use,"expected"]+e.loc[use,"residual"],(g.mag_lo+g.mag_hi)/2,g.variance)
    e=e[np.isfinite(e.s2)]
    ccd=e.drop_duplicates(["star","band"]).copy()
    ccd["native"],ccd["residual"],ccd["s2"],ccd["mjd"],ccd["plate"]="ccd",0.,ccd.reference_variance,sn.CCD_MJD,-1
    e=pd.concat([e,ccd],ignore_index=True).sort_values(["star","band"],kind="stable").reset_index(drop=True)
    i,j=esf.group_pairs(e.star.to_numpy()*2+e.band.eq("r").to_numpy())
    code=e.native.map({"ccd":0,"bj":1,"r":2,"e":3,"tp":4}).to_numpy()
    p=pd.DataFrame(dict(star=e.star.to_numpy()[i],band=e.band.to_numpy()[i],
        a=np.minimum(code[i],code[j]),b=np.maximum(code[i],code[j]),mag=e.reference.to_numpy()[i],
        dm=e.residual.to_numpy()[i]-e.residual.to_numpy()[j],se2=e.s2.to_numpy()[i]+e.s2.to_numpy()[j],
        lag=np.abs(e.mjd.to_numpy()[i]-e.mjd.to_numpy()[j]),pi=e.plate.to_numpy()[i],pj=e.plate.to_numpy()[j]))
    table=pd.read_csv(ROOT/"data/plate_native_pair_variance.csv")
    native=("ccd","bj","r","e","tp")
    p["class_id"]=-1;p["floor"],p["center"],p["limit"]=np.nan,np.nan,np.nan
    for (a,b),q in table.groupby(["native_a","native_b"]):
        use=p.a.eq(native.index(a))&p.b.eq(native.index(b))
        q=q.sort_values("mag_lo");centers=(q.mag_lo+q.mag_hi).to_numpy()/2
        pos=np.searchsorted((centers[1:]+centers[:-1])/2,p.loc[use,"mag"])
        for col,source in (("class_id","class_id"),("floor","residual_variance"),("center","clip_center"),("limit","clip_limit")):
            p.loc[use,col]=q[source].to_numpy()[pos]
    p["bin"]=np.searchsorted(sn.NULL_LAG,p.lag,side="right")-1
    p=p[p.class_id.ge(0)&p.bin.ge(0)&p.bin.lt(len(sn.NULL_LAG)-1)&(p.dm-p.center).abs().le(p.limit)].copy()
    p["before"]=p.dm**2-p.se2;p["after"]=p.before-p.floor
    limits=sn._fiducial_limits();draws=np.load(ROOT/"data/plate_native_pair_variance_bootstrap.npz")["draws"]
    rows,covariances=[],{}
    rng=np.random.default_rng(731091)
    for band in "gr":
        lo,hi=limits.loc[band];q=p[p.band.eq(band)&p.mag.between(lo,hi)]
        stats=q.groupby(["star","bin"])[["before","after"]].mean().reset_index()
        ids,si=np.unique(stats.star,return_inverse=True);nb=len(sn.NULL_LAG)-1
        counts=coo_matrix((np.ones(len(stats)),(si,stats.bin)),shape=(len(ids),nb)).tocsr()
        values=coo_matrix((stats.before,(si,stats.bin)),shape=counts.shape).tocsr()
        n=np.asarray(counts.sum(axis=0)).ravel();mean=np.asarray(values.sum(axis=0)).ravel()/np.maximum(n,1)
        boot=[]
        for start in range(0,len(draws),40):
            w=rng.poisson(1,(min(40,len(draws)-start),len(ids)))
            denom=np.asarray(w@counts);boot.append(np.divide(w@values,denom,out=np.full(denom.shape,np.nan),where=denom>0))
        boot=np.concatenate(boot)
        weight=q.groupby(["star","bin","class_id"]).size().rename("n").reset_index()
        weight["fraction"]=weight.n/weight.groupby(["star","bin"]).n.transform("sum")
        mix=weight.groupby(["bin","class_id"]).fraction.sum().unstack(fill_value=0).reindex(index=range(nb),columns=range(draws.shape[1]),fill_value=0).to_numpy()/np.maximum(n[:,None],1)
        floor=table.set_index("class_id").residual_variance.reindex(range(draws.shape[1])).to_numpy()
        after=mean-mix@floor
        varied=boot-(draws-draws.mean(axis=0))@mix.T-mix@floor
        plates=np.unique(np.r_[q.pi,q.pj]);plates=plates[plates>=0]
        jack=[]
        for plate in plates:
            leave=q[q.pi.ne(plate)&q.pj.ne(plate)].groupby(["star","bin"])[["before","after"]].mean().groupby("bin").mean().reindex(range(nb))
            jack.append(leave.to_numpy())
        jack=np.array(jack)
        centered=jack-np.nanmean(jack,axis=0)
        factor=(len(plates)-1)/len(plates)
        floor_cov=np.cov((draws-draws.mean(axis=0))@mix.T,rowvar=False)
        covariances[band]=factor*np.nan_to_num(centered[:,:,1]).T@np.nan_to_num(centered[:,:,1])+floor_cov
        before_error=np.sqrt(factor*np.nansum(centered[:,:,0]**2,axis=0))
        after_error=np.sqrt(np.diag(covariances[band]))
        for k in np.flatnonzero(n>=30):
            z=q[q.bin.eq(k)];plates=np.unique(np.r_[z.pi,z.pj]);plates=plates[plates>=0]
            valid=len(plates)>=3 and np.isfinite(jack[:,k]).all()
            if not valid:
                before_error[k]=np.nanstd(boot[:,k],ddof=1)
                after_error[k]=np.nanstd(varied[:,k],ddof=1)
            rows.append(dict(band=band,mag_lo=lo,mag_hi=hi,observed_days=np.sqrt(sn.NULL_LAG[k]*sn.NULL_LAG[k+1]),
                sf2_before=mean[k],sf2_after=after[k],error_before=before_error[k],error_after=after_error[k],
                star_bootstrap_error=np.nanstd(varied[:,k],ddof=1),
                n_stars=int(n[k]),n_pairs=len(z),n_plates=len(plates),plate_jackknife_valid=valid,lag_bin=k))
    result=pd.DataFrame(rows)
    result.to_csv(ROOT/"data/standard_star_adopted_null_structure_function.csv",index=False)
    np.savez_compressed(ROOT/"data/standard_star_adopted_null_covariance.npz",**covariances)
    print(result.to_string(index=False))


def native_floor_impact():
    import plate_completeness as pc
    import sf_model_checks as checks
    samples=list(checks.samples())
    all_ids=np.unique(np.concatenate([p.index.to_numpy(str) for _,_,_,p,_ in samples]))
    nightly=esf.clean_nightly(pc.load_completeness_lightcurves(all_ids))[0]
    redshift=pd.concat([p.z for _,_,_,p,_ in samples]).groupby(level=0).first()
    table=pd.read_csv(ROOT/"data/plate_native_pair_variance.csv")
    draws=np.load(ROOT/"data/plate_native_pair_variance_bootstrap.npz")["draws"]
    mean=table.set_index("class_id").residual_variance.reindex(range(draws.shape[1])).to_numpy()
    delta=draws-mean
    codes={"ccd":0,"bj":1,"r":2,"e":3,"tp":5}
    null=pd.read_csv(ROOT/"data/standard_star_adopted_null_structure_function.csv")
    objects={};rng=np.random.default_rng(42115)
    for (obj,band),g in nightly.groupby(["OBJID","band"],observed=True,sort=False):
        if band not in "gr":continue
        g=g.sort_values("night",kind="stable")
        t,mag,sig,vt=(g[c].to_numpy(float) for c in ("night","mag","sig","trunc_v"))
        i,j=np.triu_indices(len(g),1);observed=t[j]-t[i]
        bins=np.searchsorted(esf.EDGES,observed/(1+redshift[obj]),side="right")-1
        use=(bins>=0)&(bins<esf.NB);i,j,bins,observed=i[use],j[use],bins[use],observed[use]
        dm=mag[j]-mag[i];w=2/(vt[i]+vt[j]);code=g.pcode.to_numpy()
        moment=w*(dm**2-sig[i]**2-sig[j]**2-g[esf.FV_COLS].to_numpy()[i,code[j]])
        cls=np.full(len(i),-1,dtype=int)
        ref=g.loc[~g.sss,"mag"].median()
        for (a,b),q in table.groupby(["native_a","native_b"]):
            q=q.sort_values("mag_lo");cen=(q.mag_lo+q.mag_hi).to_numpy()/2
            k=np.searchsorted((cen[1:]+cen[:-1])/2,ref)
            use=((code[i]==codes[a])&(code[j]==codes[b]))|((code[i]==codes[b])&(code[j]==codes[a]))
            cls[use]=q.class_id.iloc[k]
        z=null[null.band.eq(band)].sort_values("observed_days")
        systematic=np.interp(np.log10(np.maximum(observed,1)),np.log10(z.observed_days),z.sf2_after)*w
        systematic[(code[i]==0)&(code[j]==0)]=0.
        weights=np.zeros((3,esf.NB,draws.shape[1]));moments=np.full((3,esf.NB),np.nan);shift=moments.copy()
        for sign,use in enumerate((np.ones(len(dm),bool),dm<0,dm>0)):
            count=np.bincount(bins[use],minlength=esf.NB)
            moments[sign]=np.divide(np.bincount(bins[use],weights=moment[use],minlength=esf.NB),count,out=np.full(esf.NB,np.nan),where=count>0)
            shift[sign]=np.divide(np.bincount(bins[use],weights=systematic[use],minlength=esf.NB),count,out=np.full(esf.NB,np.nan),where=count>0)
            take=use&(cls>=0)
            np.add.at(weights[sign],(bins[take],cls[take]),w[take]/count[bins[take]])
        objects[(str(obj),band)]=(moments,weights,shift)
    rows=[]
    for band,name,acc,prop,kernel in samples:
        nb=acc.shape[1];bundle=[objects[(str(obj),band)] for obj in prop.index]
        moments=np.array([v[0][:,:nb] for v in bundle]);weights=np.array([v[1][:,:nb] for v in bundle]);shifts=np.array([v[2][:,:nb] for v in bundle])
        target=np.nanmean(moments,axis=0);n=np.isfinite(moments).sum(axis=0)
        mix=weights.sum(axis=0)/np.maximum(n[:,:,None],1)
        varied=target[None,:,:]-np.einsum("dc,skc->dsk",delta,mix)
        sf=np.sqrt(np.where(target>0,target,np.nan));varied_sf=np.sqrt(np.where(varied>0,varied,np.nan))
        np.testing.assert_allclose(target[0],esf.object_second_moments(acc).mean(axis=0),atol=1e-8,rtol=1e-7)
        boot=[]
        for _ in range(400):boot.append(np.nanmean(moments[rng.integers(0,len(moments),len(moments))],axis=0))
        boot=np.array(boot);boot_sf=np.sqrt(np.where(boot>0,boot,np.nan))
        systematic=np.nanmean(shifts,axis=0)
        for k in range(nb):
            for s,label in enumerate(("all","brightening","dimming")):
                rows.append(dict(band=band,sample=name,statistic=label,lag_days=esf.CENTERS[k],n_objects=int(n[s,k]),
                    value=sf[s,k],floor_sigma=np.nanstd(varied_sf[:,s,k],ddof=1),quasar_sigma=np.nanstd(boot_sf[:,s,k],ddof=1),
                    lag_null_shift=np.sqrt(target[s,k]-systematic[s,k])-sf[s,k] if target[s,k]>systematic[s,k] else np.nan))
            beta=(sf[1,k]-sf[2,k])/sf[0,k]
            floor_beta=(varied_sf[:,1,k]-varied_sf[:,2,k])/varied_sf[:,0,k]
            boot_beta=(boot_sf[:,1,k]-boot_sf[:,2,k])/boot_sf[:,0,k]
            rows.append(dict(band=band,sample=name,statistic="asymmetry",lag_days=esf.CENTERS[k],n_objects=int(n[0,k]),
                value=beta,floor_sigma=np.nanstd(floor_beta,ddof=1),quasar_sigma=np.nanstd(boot_beta,ddof=1),lag_null_shift=np.nan))
    result=pd.DataFrame(rows);result.to_csv(ROOT/"data/plate_native_floor_impact.csv",index=False)
    print(result.groupby(["band","statistic"])[["floor_sigma","quasar_sigma","lag_null_shift"]].agg(["min","max"]).to_string())


def native_sed_impact():
    import plate_completeness as pc
    import sf_model_checks as checks
    selected=[q for q in checks.samples() if q[1]=="fiducial"]
    ids=np.unique(np.concatenate([p.index.to_numpy(str) for _,_,_,p,_ in selected]))
    nightly=esf.clean_nightly(pc.load_completeness_lightcurves(ids))[0]
    epochs=pd.read_parquet(ROOT/"data/plate_native_quasar_epochs.parquet")
    epochs["OBJID"]=epochs.OBJID.astype(str)
    epochs=epochs.set_index(["OBJID","band","PLATEID"])
    rows=[];rng=np.random.default_rng(7818)
    for source,column,flag in (("native","delta_qsogen","valid"),("observed","delta_observed","quality")):
        control=pd.read_csv(ROOT/f"data/quasar_{source}_sed_control.csv")
        control["OBJID"]=control.OBJID.astype(str)
        control=control[control[flag]&np.isfinite(control[column])].set_index(["OBJID","native"])
        for band,_,acc,prop,kernel in selected:
            nb=acc.shape[1];valid=[];alternatives=[]
            for k,obj in enumerate(prop.index.astype(str)):
                g=nightly[nightly.OBJID.eq(obj)&nightly.band.eq(band)].copy()
                q=epochs.reindex(pd.MultiIndex.from_arrays([g.loc[g.sss,"OBJID"],g.loc[g.sss,"band"],g.loc[g.sss,"plate"]]))
                terms=control.reindex(pd.MultiIndex.from_arrays([q.index.get_level_values(0),q.native]))[column]
                if terms.isna().any():continue
                pogson=q.native_mag.to_numpy()+q.sed_term.to_numpy()+terms.to_numpy()
                softening=9e-11 if band=="g" else 1.2e-10
                g.loc[g.sss,"mag"]=-2.5/np.log(10)*(np.arcsinh(10**(-.4*pogson)/(2*softening))+np.log(softening))
                alternative=esf.accumulate(g,np.array([obj]),prop.z,edges=esf.EDGES[:nb+1],bands=[band],resid=True)[0,0]
                np.testing.assert_array_equal(alternative[:,0],acc[k,:,0])
                valid.append(k);alternatives.append(esf.object_second_moments(alternative[None])[0])
            if not valid:continue
            base=esf.object_second_moments(acc[np.array(valid)]);alternative=np.array(alternatives)
            point=np.sqrt(base.mean(axis=0));other=np.sqrt(alternative.mean(axis=0))
            w=rng.multinomial(len(valid),np.full(len(valid),1/len(valid)),size=800)
            delta=np.sqrt(np.maximum(w@alternative/len(valid),0))-np.sqrt(np.maximum(w@base/len(valid),0))
            bounds=np.percentile(delta,[16,84],axis=0)
            for k in range(nb):
                rows.append(dict(control=source,band=band,n_fiducial=len(prop),n_common=len(valid),lag_days=esf.CENTERS[k],
                    composite_sf_mag=point[k],alternative_sf_mag=other[k],difference_mag=other[k]-point[k],
                    difference_lo=bounds[0,k],difference_hi=bounds[1,k]))
    result=pd.DataFrame(rows);result.to_csv(ROOT/"data/plate_native_sed_sensitivity.csv",index=False)
    print(result.groupby(["control","band"]).agg(n_common=("n_common","first"),max_abs_difference=("difference_mag",lambda x:x.abs().max())).to_string())


def native_variant(variant,adopted):
    global NATIVE_VARIANT,NATIVE_WORK
    if variant is None and adopted.intersection(sys.argv) and (ROOT/"data/plate_native_calibration.json").exists():
        variant="effective"
    if variant is not None:
        if variant not in ("bounded","energy","effective"):
            raise ValueError("Unknown native calibration sensitivity")
        NATIVE_VARIANT,NATIVE_WORK=variant,NATIVE_BASE/variant
    if adopted.intersection(sys.argv) and NATIVE_VARIANT!="effective":
        raise ValueError("Adopted native diagnostics require --native-variant effective")


if __name__ == "__main__":
    esf.check_flags(__file__)
    native_variant(sys.argv[sys.argv.index("--native-variant")+1] if "--native-variant" in sys.argv else None,
                   {"--native-floor-impact","--native-sed-impact"})
    if "--native-fit" in sys.argv:
        native_calibration()
    elif "--native-quasars" in sys.argv:
        native_quasars()
    elif "--native-validation" in sys.argv:
        native_validation()
    elif "--native-noise" in sys.argv:
        native_noise()
    elif "--native-publish" in sys.argv:
        native_publish()
    elif "--native-floor-impact" in sys.argv:
        native_floor_impact()
    elif "--native-sed-impact" in sys.argv:
        native_sed_impact()
