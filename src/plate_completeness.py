from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.stats import norm

from assemble_lightcurves import (
    _pava_decreasing, ccd_reference_mags, ccd_truncation_curve, field_coords,
    plate_centres, truncation_curve, sss_raw, assign_group_ids_by_sky, match_groups_to_catalog,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SERIES = {"SERC-J/EJ": "g", "SERC-R/AAO-R": "r", "POSSI-E(S)": "r"}
PRIMARY = {"g": "SERC-J/EJ", "r": "POSSI-E(S)"}
MARGINS = (0.25, 0.5, 0.75, 1.0, 1.25)
ADOPTED_SELECTION = {"g": "all_cut_025", "r": "all_cut_025"}
BRIGHT_BOUND = 17.5
BLAZAR_RADIUS = 2.0
INSET = 0.02


def blazars(cat):
    bz = Table.read(DATA / "bzcat5.dat", format="ascii.cds", readme=str(DATA / "bzcat5_readme.txt")).to_pandas()
    sky = SkyCoord(15*(bz.RAh+bz.RAm/60+bz.RAs/3600), np.where(bz["DE-"] == "-", -1, 1)*(bz.DEd+bz.DEm/60+bz.DEs/3600), unit="deg")
    i, sep, _ = SkyCoord(cat.RA, cat.DEC, unit="deg").match_to_catalog_sky(sky)
    m = sep.arcsec <= BLAZAR_RADIUS
    out = pd.DataFrame({"OBJID": cat.objectId[m].astype(str), "z": cat.Z_DR16Q[m], "bzcat_name": bz.Name.to_numpy()[i[m]],
                        "bzcat_class": bz.Class.to_numpy()[i[m]], "bzcat_z": bz.z.to_numpy()[i[m]], "separation_arcsec": sep.arcsec[m]})
    out.to_csv(DATA / "bzcat5_stripe82_matches.csv", index=False)
    return out


def catalog():
    cat = pd.read_parquet(DATA / "S82/Catalog.parquet")
    cat = cat[~cat.objectId.astype(str).isin(blazars(cat).OBJID)]
    cat["OBJID"] = cat["objectId"].astype(str)
    cat["query_key"] = [f"{ra:.6f}_{dec:.6f}" for ra, dec in zip(cat.RA, cat.DEC)]
    with fits.open(DATA / "dr16q_prop_May01_2024.fits", memmap=True) as h:
        d = h[1].data
        prop = pd.DataFrame({n: d[n].astype(np.int64) for n in ("PLATE", "MJD", "FIBERID")}
                            | {n: d[n].astype(float) for n in ("RA", "DEC", "Z_DR16Q", "LOGLBOL")})
    return cat.merge(prop[["PLATE", "MJD", "FIBERID", "LOGLBOL"]],
                     on=["PLATE", "MJD", "FIBERID"], validate="one_to_one"), prop


def inputs():
    cat, prop = catalog()
    lc = pd.read_parquet(DATA / "S82/total_lightcurves.parquet",
                         filters=[("band", "in", ["g", "r"])])
    lc["OBJID"] = lc.OBJID.astype(str)
    modern = lc.loc[~lc.survey.str.startswith("sss")].copy()
    modern["night"] = np.floor(modern.time).astype(np.int64)
    nights = modern.groupby(["OBJID", "band", "survey", "night"], observed=True).mag.mean()
    stats = nights.groupby(["OBJID", "band", "survey"]).agg(["mean", "median", "count"]).reset_index()
    ref = stats.groupby(["OBJID", "band"])["mean"].mean().unstack().reindex(cat.OBJID)
    original_median = modern.groupby(["OBJID", "band"]).mag.median()
    plates = lc.loc[lc.survey.str.startswith("sss")].copy()
    return cat, prop, stats, ref, original_median, plates


def opportunities(cat, prop, ref, plates):
    parent = prop[((prop.RA > 300) | (prop.RA < 60)) & (prop.DEC > -1.3) & (prop.DEC < 1.3)]
    query_keys = set(f"{ra:.6f}_{dec:.6f}" for ra, dec in zip(parent.RA, parent.DEC))
    key_index = dict(zip(cat.query_key, cat.index))
    raw = []
    with h5py.File(DATA / "wu_qso.hdf5") as h:
        seen_keys = set(h)
        if not seen_keys <= query_keys or not set(cat.query_key) <= query_keys:
            raise ValueError("plate query coordinates do not match the documented parent")
        for key in h:
            if key not in key_index:
                continue
            k = key_index[key]
            a = h[key][:]
            sep = 3600 * np.hypot(((a["ra"] - cat.RA[k] + 180) % 360 - 180)
                                  * np.cos(np.deg2rad(cat.DEC[k])), a["dec"] - cat.DEC[k])
            raw.extend((k, int(pl), float(dd)) for pl, dd in zip(a["plateID"], sep))
    raw = pd.DataFrame(raw, columns=["parent_index", "plate", "separation_arcsec"])
    raw = raw.groupby(["parent_index", "plate"]).separation_arcsec.min()
    groups = assign_group_ids_by_sky(sss_raw())
    mapping = match_groups_to_catalog(groups, str(DATA / "S82/Catalog.parquet"))
    matched = groups.drop(columns=["OBJID", "z"], errors="ignore").merge(mapping, on="GROUP_ID")
    matched = set(zip(matched.OBJID.astype(str), matched.PLATEID.astype(int)))
    kept = set(zip(plates.OBJID, plates.plate.astype(int)))
    flagged = set(zip(plates.loc[~plates.calib_ok, "OBJID"], plates.loc[~plates.calib_ok, "plate"].astype(int)))
    rows = []
    for p, k, edge in footprint(cat, pd.read_csv(DATA / "plate_native_calibration.csv")):
        ind = pd.MultiIndex.from_arrays([k, np.full(len(k), p.plate)])
        sep = raw.reindex(ind).to_numpy()
        obj = cat.OBJID.iloc[k].to_numpy()
        rows.append(pd.DataFrame({"OBJID": obj, "plate": p.plate, "series": p.survey,
                                  "band": SERIES[p.survey], "reference_mag": ref[SERIES[p.survey]].iloc[k].to_numpy(),
                                  "z": cat.Z_DR16Q.iloc[k].to_numpy(), "edge_distance_deg": edge,
                                  "interior": edge >= INSET, "archive_detection": np.isfinite(sep),
                                  "matched_1arcsec": [(o, p.plate) in matched for o in obj],
                                  "individual_within_1arcsec": sep <= 1, "separation_arcsec": sep,
                                  "kept_epoch": [(o, p.plate) in kept for o in obj],
                                  "quality_flagged": [(o, p.plate) in flagged for o in obj]}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(DATA / "plate_completeness_opportunities.parquet", index=False)
    notebook = json.loads((ROOT / "src/query.ipynb").read_text())
    outputs = "".join("".join(o.get("text", [])) for c in notebook["cells"] for o in c.get("outputs", []))
    if "39827/39827" not in outputs:
        raise ValueError("documented full query completion is missing")
    provenance = {
        "wu_shen_all": len(prop), "queried_stripe82": len(parent), "positive_query_groups": len(seen_keys),
        "no_query_detection": len(parent)-len(seen_keys), "modern_parent": len(cat),
        "modern_parent_positive_query": int(cat.query_key.isin(seen_keys).sum()),
        "modern_parent_no_query_detection": int((~cat.query_key.isin(seen_keys)).sum()),
        "modern_parent_without_g_mean": int(ref.g.isna().sum()),
        "modern_parent_without_r_mean": int(ref.r.isna().sum()),
        "documented_query_errors": int("Failed permanently" in outputs or "Retrying" in outputs),
        "plate_opportunities_stellar_box": len(out),
        "plate_opportunities_interior": int(out.interior.sum()),
        "footprint_inset_deg": INSET,
    }
    pd.DataFrame(list(provenance.items()), columns=["quantity", "value"]).to_csv(
        DATA / "plate_completeness_provenance.csv", index=False)
    return out


def footprint(cat, zp):
    ctr = plate_centres()
    for p in zp[zp.survey.isin(SERIES)].itertuples():
        xi, eta = field_coords(np.full(len(cat), p.plate), cat.RA.to_numpy(), cat.DEC.to_numpy(), ctr)
        edge = np.minimum.reduce([xi-p.xi_lo, p.xi_hi-xi, eta-p.eta_lo, p.eta_hi-eta])
        k = np.flatnonzero(edge >= 0)
        yield p, k, edge[k]


def binned_curve(mag, value, keep, width=0.25, min_n=20, start=18.5, level=0.9, monotone=True):
    rows = []
    edges = np.arange(17.5, 23.51, width)
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = keep & np.isfinite(mag) & np.isfinite(value) & (mag >= lo) & (mag < hi)
        if k.sum() >= min_n:
            rows.append((0.5*(lo+hi), int(k.sum()), float(value[k].mean()),
                         float(value[k].std(ddof=1)/np.sqrt(k.sum()))))
    a = np.array(rows)
    fit = _pava_decreasing(np.clip(a[:, 2], 0, 1), a[:, 1]) if monotone else np.clip(a[:, 2], 0, 1)
    j = np.flatnonzero((a[:, 0] > start) & (fit < level))[0]
    limit = np.interp(level, fit[j-1:j+1][::-1], a[j-1:j+1, 0][::-1])
    return a, fit, limit


def band_statistics(cat, stats, ref, interior, band, k=25, window=(17.75, 18.75)):
    mag = ref[band].to_numpy()
    co = np.column_stack([(cat.RA.to_numpy()+180) % 360-180, cat.DEC])
    group = interior[interior.series == PRIMARY[band]].groupby("OBJID")
    inside = cat.OBJID.isin(group.size().index).to_numpy()
    detected = group.kept_epoch.any().reindex(cat.OBJID).fillna(False).to_numpy(bool)
    values = {PRIMARY[band]: (detected.astype(float), inside)}
    for name in ("sdss", "ps1", "ztf"):
        n = stats[(stats.band == band) & (stats.survey == name)].set_index("OBJID")["count"].reindex(cat.OBJID).fillna(0).to_numpy()
        bright = (mag >= window[0]) & (mag < window[1]) & (n > 0)
        neighbor = cKDTree(co[bright]).query(co, k=min(k, int(bright.sum())))[1]
        values[name] = (n/n[bright][neighbor].mean(axis=1), np.ones(len(cat), bool))
    return mag, inside, detected, values


def sample_tables(cat, stats, ref, opp):
    all_rows, curves, sweep, levels = [], [], [], []
    interior = opp[opp.interior]
    for band in "gr":
        mag, inside, detected, values = band_statistics(cat, stats, ref, interior, band)
        target = cat.Z_DR16Q.to_numpy() >= 0.5
        sv = PRIMARY[band]
        out = pd.DataFrame({"OBJID": cat.OBJID, "band": band, "reference_mag": mag,
                            "z": cat.Z_DR16Q, "loglbol": cat.LOGLBOL,
                            "plate_opportunity": inside, "plate_detected": detected})
        limits = []
        for name, (value, valid) in values.items():
            if name != sv:
                out[f"{name}_epoch_ratio"] = value
            a, monotone, limit = binned_curve(mag, value, target & valid)
            limits.append(limit)
            levels.append(dict(band=band, survey=name, limit90=limit,
                               kind="plate_object_recovery" if name == sv else "ccd_local_epoch_ratio"))
            recovery = np.interp(mag, a[:, 0], monotone)
            if name == sv:
                out["plate_recovery_fraction"] = recovery
            for row, fitted in zip(a, monotone):
                curves.append(dict(band=band, survey=name, mag_center=row[0], n_objects=int(row[1]),
                                   recovery=row[2], recovery_err=row[3], monotone_recovery=fitted,
                                   limit90=limit, kind=levels[-1]["kind"], stage="kept_epoch"))
        for margin in MARGINS:
            tag = f"{int(100*margin):03d}"
            for kind, limit in [("plate", limits[0]), ("all", min(limits))]:
                keep = inside & target & (mag >= BRIGHT_BOUND) & (mag <= limit-margin)
                out[f"{kind}_cut_{tag}"] = keep
                row = dict(band=band, selection=f"{kind}_cut_{tag}", margin_mag=margin,
                           mag_min=BRIGHT_BOUND, mag_max=limit-margin, n_parent=int(keep.sum()),
                           n_detected=int(detected[keep].sum()), recovery=float(detected[keep].mean()),
                           median_z=float(cat.Z_DR16Q[keep].median()), median_loglbol=float(cat.LOGLBOL[keep].median()))
                for survey in ("sdss", "ps1", "ztf"):
                    row[f"{survey}_epoch_ratio"] = out.loc[keep, f"{survey}_epoch_ratio"].mean()
                sweep.append(row)
        out["plate_limit90"], out["ccd_limit90"] = limits[0], min(limits[1:])
        out["adopted_keep"] = False
        all_rows.append(out)
    for sv, band in SERIES.items():
        g = interior[interior.series == sv]
        for weighting in ("object", "opportunity"):
            t = g.groupby("OBJID").agg(reference_mag=("reference_mag", "first"), z=("z", "first"),
                                        archive_detection=("archive_detection", "any"),
                                        matched_1arcsec=("matched_1arcsec", "any"), kept_epoch=("kept_epoch", "any")) if weighting == "object" else g
            for stage in ("archive_detection", "matched_1arcsec", "kept_epoch"):
                a, monotone, limit = binned_curve(t.reference_mag.to_numpy(), t[stage].to_numpy(float), t.z.to_numpy() >= 0.5)
                for row, fitted in zip(a, monotone):
                    curves.append(dict(band=band, survey=sv, mag_center=row[0], n_objects=int(row[1]),
                                       recovery=row[2], recovery_err=row[3], monotone_recovery=fitted,
                                       limit90=limit, kind=f"plate_{weighting}_recovery", stage=stage))
    sample = pd.concat(all_rows, ignore_index=True)
    sample.to_parquet(DATA / "plate_completeness_sample.parquet", index=False)
    pd.DataFrame(curves).drop_duplicates().to_csv(DATA / "plate_completeness_recovery.csv", index=False)
    pd.DataFrame(sweep).to_csv(DATA / "plate_completeness_cut_sweep.csv", index=False)
    pd.DataFrame(levels).to_csv(DATA / "plate_completeness_limits.csv", index=False)
    return sample


def correction_components(original_median):
    cat = pd.read_parquet(DATA / "S82/Catalog.parquet", columns=["objectId"]).assign(OBJID=lambda d: d.objectId.astype(str))
    raw = pd.read_parquet(DATA / "S82/dr16s82_sdssLCRaw.parquet",
                          columns=["objectId", "mjd", "filterID", "psMag", "psMagErr_p3"])
    raw = raw[raw.objectId.isin(cat.objectId) & raw[["mjd", "psMag", "psMagErr_p3"]].notna().all(axis=1)].copy()
    raw["OBJID"] = raw.objectId.astype(str)
    raw["band"] = raw.filterID.map(dict(enumerate("ugriz")))
    raw = raw.rename(columns={"psMag": "mag"})
    raw["mag"] = raw.mag.astype(float)
    raw["survey"] = "sdss"
    reference = ccd_reference_mags(raw)
    rows = []
    for (sv, band), g in ccd_truncation_curve().groupby(["survey", "band"]):
        if band not in "gr":
            continue
        rv = reference.reindex(pd.MultiIndex.from_arrays([cat.OBJID, np.repeat(band, len(cat))])).to_numpy()
        shift = np.interp(rv, g.ref_mag, g.offset_mag)
        rows.append(pd.DataFrame({"OBJID": cat.OBJID, "band": band, "survey": sv,
                                  "reference_used": rv, "mag_add_no_dropout": shift,
                                  "component": "ccd_mean", "calibration_offset_retained": 0.0}))
        if sv == "sdss":
            before = raw[raw.band == band].groupby("OBJID").mag.median().reindex(cat.OBJID).to_numpy()
            after = pd.read_parquet(DATA / "S82/total_lightcurves.parquet", columns=["OBJID", "mag"],
                                     filters=[("survey", "==", "sdss"), ("band", "==", band)]).groupby("OBJID").mag.median().reindex(cat.OBJID).to_numpy()
            if np.nanmax(np.abs(before-after-shift)) > 1e-9:
                raise ValueError("CCD correction reversal does not reproduce raw SDSS medians")
    for band, survey in (("g", "sss"), ("r", "sss"), ("r", "sss_possi")):
        rv = original_median.reindex(pd.MultiIndex.from_arrays([cat.OBJID, np.repeat(band, len(cat))])).to_numpy()
        rows.append(pd.DataFrame({"OBJID": cat.OBJID, "band": band, "survey": survey,
                                  "reference_used": rv, "mag_add_no_dropout": 0.,
                                  "component": "native_sed", "calibration_offset_retained": 0.}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(DATA / "detection_correction_components.parquet", index=False)
    return out


def footprint_check(sample, opp):
    rows = []
    for sv, g in opp.groupby("series"):
        for selection in sorted(set(ADOPTED_SELECTION.values())):
            ids = set(sample.loc[(sample.band == g.band.iloc[0]) & sample[selection], "OBJID"])
            for inset in (0.0, INSET, 0.05):
                x = g[(g.edge_distance_deg >= inset) & g.OBJID.isin(ids)]
                grouped = x.groupby("OBJID").kept_epoch.any()
                rows.append(dict(series=sv, selection=selection, inset_deg=inset,
                                 n_objects=len(grouped), n_opportunities=len(x), object_recovery=grouped.mean(),
                                 epoch_recovery=x.kept_epoch.mean(), archive_recovery=x.archive_detection.mean(),
                                 matched_recovery=x.matched_1arcsec.mean()))
    pd.DataFrame(rows).to_csv(DATA / "plate_completeness_footprint_check.csv", index=False)


def cut_recovery_stages(sample, opp):
    sweep = pd.read_csv(DATA / "plate_completeness_cut_sweep.csv")
    for i, row in sweep.iterrows():
        chosen = sample[(sample.band == row.band) & sample[row.selection]]
        p = opp[(opp.series == PRIMARY[row.band]) & opp.interior & opp.OBJID.isin(chosen.OBJID)]
        sweep.loc[i, "n_plate_opportunities"] = len(p)
        for stage in ("archive_detection", "matched_1arcsec", "kept_epoch"):
            sweep.loc[i, f"n_objects_{stage}"] = int(p.groupby("OBJID")[stage].any().sum())
            sweep.loc[i, f"n_epochs_{stage}"] = int(p[stage].sum())
            sweep.loc[i, f"epoch_fraction_{stage}"] = p[stage].mean()
    sweep.to_csv(DATA / "plate_completeness_cut_sweep.csv", index=False)


def load_completeness_lightcurves(objids=None, remove_dropout=True):
    filters = [("band", "in", ["g", "r"])]
    if objids is not None:
        filters.append(("OBJID", "in", list(map(str, objids))))
    df = pd.read_parquet(DATA / "S82/total_lightcurves.parquet", filters=filters)
    df["OBJID"] = df.OBJID.astype(str)
    if remove_dropout:
        terms = pd.read_parquet(DATA / "detection_correction_components.parquet")
        shift = terms.set_index(["OBJID", "band", "survey"]).mag_add_no_dropout.reindex(
            pd.MultiIndex.from_frame(df[["OBJID", "band", "survey"]])).to_numpy()
        if not np.isfinite(shift).all():
            raise ValueError("missing authoritative detection-correction component")
        df["mag"] = df.mag.to_numpy()+shift
        df["trunc_v"] = 1.0
    df.attrs["photometry"] = "no_dropout" if remove_dropout else "production"
    return df


def _property_weights(features, cells, selected, counts, target, theta=None):
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    target_weights = counts * target
    target_weights /= target_weights.sum()
    goal = target_weights @ features
    cell_mass = np.bincount(cells, weights=target_weights)
    active = np.flatnonzero(selected & (counts > 0))
    x, c = features[active], cells[active]
    log_count = np.log(counts[active])
    parts = [np.flatnonzero(c == j) for j in range(len(cell_mass))]

    def objective(coef, weights_only=False):
        weights = np.zeros(len(x))
        value = -coef @ goal
        for mass, indices in zip(cell_mass, parts):
            if mass == 0:
                continue
            h = log_count[indices] + x[indices] @ coef
            z = logsumexp(h)
            value += mass*z
            weights[indices] = mass*np.exp(h-z)
        return weights if weights_only else (value, weights @ x-goal)

    fit = minimize(objective, np.zeros(features.shape[1]) if theta is None else theta,
                   method="BFGS", jac=True, options={"gtol": 1e-8, "maxiter": 250})
    weight = np.zeros(len(features))
    weight[active] = objective(fit.x, True)
    error = np.max(np.abs(weight @ features-goal))
    if not np.isfinite(error) or error > 2e-6:
        raise ValueError(f"property balance failed: {error:.3g}; {fit.message}")
    return weight, fit.x, error


def candidate_accumulator():
    import hashlib
    import inspect
    import ensemble_structure_function as esf

    with np.load(esf.CACHE / "object_first/object_accumulator.npz") as f:
        ids, acc = f["ids"].astype(str), f["acc"]
        if int(f["completed"]) != len(ids):
            raise ValueError("incomplete object accumulator")
        signature = str(f["signature"])
    sources = ["data/detection_correction_components.parquet", "data/plate_native_pair_variance.csv"]
    kernels = [esf.accumulate, esf.clean_nightly, esf.condense_nightly, esf.reject_outliers,
               esf.floor_variance, load_completeness_lightcurves]
    stamp = (ROOT / esf.LC_PARQUET).stat()
    code = "".join(inspect.getsource(f) for f in kernels)+f"{stamp.st_size}:{stamp.st_mtime_ns}"
    expected = hashlib.sha256(b"".join((ROOT/p).read_bytes() for p in sources)
                              +ids.astype("U").tobytes()+code.encode()).hexdigest()
    if signature != expected:
        raise ValueError("object accumulator signature differs from current inputs")
    return ids, acc, signature


def property_support(a, s, x, size=0.5, minimum=5):
    names = [f"all_cut_{int(100*m):03d}" for m in MARGINS]
    covered = (a[..., 0] > 0).all(axis=1)
    masks = np.stack([covered & s[n].fillna(False).to_numpy(bool) for n in names], axis=1)
    valid = np.isfinite(x).all(axis=1)
    cell = np.floor(np.nan_to_num((x-[6, -5])/size)).astype(int) @ [100, 1]
    occupancy = np.stack([np.bincount(cell[m & valid], minlength=cell.max()+1) for m in masks.T])
    common = np.flatnonzero(occupancy.min(axis=0) >= minimum)
    return names, masks, valid, cell, common


def controlled(a, s, x, z, bi, n_boot, size=0.5, minimum=5, redshift=False, control=True):
    import ensemble_structure_function as esf
    from fit_structure_function import _object_amplitude

    names, masks, valid, cell, common = property_support(a, s, x, size, minimum)
    if control and not len(common):
        raise ValueError("no property cell holds the minimum under every cut")
    take = masks[:, 0] & valid & np.isin(cell, common) if control else masks[:, 0]
    ix = np.flatnonzero(take)
    c = np.searchsorted(common, cell[ix])
    mm = masks[ix]
    target = mm[:, -1]
    prop = np.column_stack([x, z])[ix] if redshift else x[ix]
    standardized = (prop-prop[target].mean(axis=0))/prop[target].std(axis=0)
    features = np.column_stack([standardized, standardized**2, standardized[:, 0]*standardized[:, 1]])
    values = esf.object_second_moments(a[ix])
    fits = [_property_weights(features, c, mm[:, j], np.ones(len(ix)), target) if control
            else (mm[:, j]/mm[:, j].sum(), None, 0.0) for j in range(len(names))]
    point = _object_amplitude(np.stack([f[0] for f in fits]) @ values)
    reps = np.empty((n_boot, len(names), a.shape[1]))
    rng = np.random.default_rng(928+bi)
    max_error = 0.0
    for b in range(n_boot):
        counts = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix))).astype(float)
        for j in range(len(names)):
            if control and j < len(names)-1:
                w, _, error = _property_weights(features, c, mm[:, j], counts, target, fits[j][1])
                reps[b, j] = _object_amplitude(w @ values)
                max_error = max(max_error, error)
            else:
                w = counts*mm[:, j]
                reps[b, j] = _object_amplitude(w @ values/w.sum())
    return dict(names=names, masks=masks, take=take, ix=ix, c=c, common=common, mm=mm,
                fits=fits, point=point, reps=reps, max_error=max_error)


def max_t(point, reps, j, others, long):
    delta = reps[:, j, long][:, None]-reps[:, others][:, :, long]
    se = delta.std(axis=0, ddof=1)
    null = np.max(np.abs((delta-delta.mean(axis=0))/se), axis=(1, 2))
    statistic = float(np.max(np.abs(point[j, long]-point[others][:, long])/se))
    return statistic, np.percentile(null, 95), np.mean(null >= statistic), delta, se


def property_control(n_boot=2000):
    import ensemble_structure_function as esf
    from fit_structure_function import _object_amplitude

    ids, acc, signature = candidate_accumulator()
    sample = pd.read_parquet(DATA / "plate_completeness_sample.parquet")
    x = esf.load_properties(ids).reindex(ids).to_numpy(float)
    native = pd.read_csv(DATA / "structure_function_completeness_sweep.csv")
    rows, summary, differences, checks, weight_rows, cell_rows = [], [], [], [], [], []
    for bi, band in enumerate("gr"):
        s = sample[sample.band.eq(band)].set_index("OBJID").reindex(ids)
        endpoint = 13 if band == "g" else 14
        a = acc[:, bi, :endpoint]
        names, masks, _, _, common = property_support(a, s, x)
        if len(common) < 4:
            checks.append(dict(band=band, status="insufficient_property_support", n_common_cells=len(common),
                               n_brightest=int(masks[:, -1].sum()), required_common_cells=4))
            continue
        r = controlled(a, s, x, s.z.to_numpy(float), bi, n_boot)
        names, masks, take, ix, c, common, mm, point, reps = (
            r[k] for k in ("names", "masks", "take", "ix", "c", "common", "mm", "point", "reps"))
        target = mm[:, -1]
        prop = x[ix]
        for j, name in enumerate(names):
            w, coef, error = r["fits"][j]
            for stage, use, ww in [("unweighted", masks[:, j], None),
                                   ("common_support", take & masks[:, j], None),
                                   ("property_control", take, w)]:
                p = x[use] if ww is None else prop
                zz = s.z.to_numpy(float)[use] if ww is None else s.z.to_numpy(float)[ix]
                good = np.isfinite(p).all(axis=1)
                ww = np.ones(good.sum())/good.sum() if ww is None else ww[good]
                p, zz = p[good], zz[good]
                pm = ww @ p
                summary.append(dict(band=band, selection=name, weighting=stage,
                                    n_objects=int((ww > 0).sum()), effective_n=1/(ww @ ww),
                                    mean_logmbh=pm[0], mean_logedd=pm[1],
                                    std_logmbh=np.sqrt(ww @ (p[:, 0]-pm[0])**2),
                                    std_logedd=np.sqrt(ww @ (p[:, 1]-pm[1])**2),
                                    mean_z=ww @ zz,
                                    n_common_cells=len(common), maximum_balance_residual=error,
                                    accumulator_signature=signature))
            for k, old_cell in enumerate(common):
                cell_rows.append(dict(band=band, selection=name, cell=k,
                                      logmbh_lo=6+.5*(old_cell//100), logmbh_hi=6+.5*(old_cell//100+1),
                                      logedd_lo=-5+.5*(old_cell % 100), logedd_hi=-5+.5*(old_cell % 100+1),
                                      n_objects=int(np.sum(mm[:, j] & (c == k))),
                                      target_fraction=float(np.mean(c[target] == k)),
                                      weighted_fraction=float(w[c == k].sum())))
            weight_rows.extend(dict(OBJID=o, band=band, selection=name, weight=v, cell=int(k))
                               for o, v, k in zip(ids[ix], w, c) if v > 0)
        bounds = np.percentile(reps, [16, 84], axis=0)
        long = np.flatnonzero(esf.CENTERS[:endpoint] >= 2371)
        for j, name in enumerate(names):
            old = native[native.band.eq(band) & native.selection.eq(name)].sort_values("lag_center_days")
            current = _object_amplitude(esf.ensemble_second_moment(a[masks[:, j]]))
            if not np.allclose(old.sf_mag, current, rtol=1e-10, atol=1e-10):
                raise ValueError(f"{band} {name}: current moments differ from native sweep")
            w = r["fits"][j][0]
            for k in range(endpoint):
                rows.append(dict(band=band, selection=name, lag_center_days=esf.CENTERS[k],
                                 sf_mag=point[j, k], sf_lo_mag=bounds[0, j, k], sf_hi_mag=bounds[1, j, k],
                                 native_sf_mag=current[k], n_native=int(masks[:, j].sum()),
                                 n_objects=int(mm[:, j].sum()), effective_n=1/(w @ w),
                                 n_common_cells=len(common), bootstrap_draws=n_boot,
                                 maximum_bootstrap_balance_residual=r["max_error"]))
            for l in range(j+1, len(names)):
                statistic, threshold, p, delta, se = max_t(point, reps, j, [l], long)
                checks.append(dict(band=band, selection=name, comparison=names[l],
                                   maximum_standardized_difference=statistic,
                                   threshold95=threshold, simultaneous_p=p, bootstrap_draws=n_boot,
                                   n_objects=int(mm[:, j].sum()), n_comparison=int(mm[:, l].sum())))
                for q, k in enumerate(long):
                    lo, hi = np.percentile(delta[:, 0, q], [16, 84])
                    differences.append(dict(band=band, selection=name, comparison=names[l],
                                            lag_center_days=esf.CENTERS[k], difference_mag=point[j, k]-point[l, k],
                                            difference_lo_mag=lo, difference_hi_mag=hi,
                                            difference_se_mag=se[0, q], bootstrap_draws=n_boot))
    for name, data in [("properties", summary), ("property_control", rows), ("property_differences", differences),
                       ("property_stability", checks), ("property_weights", weight_rows), ("property_cells", cell_rows)]:
        pd.DataFrame(data).to_csv(DATA / f"structure_function_completeness_{name}.csv", index=False)
    footprint_check(sample, pd.read_parquet(DATA / "plate_completeness_opportunities.parquet"))
    print(pd.DataFrame(checks).to_string(index=False))


def main():
    cat, prop, stats, ref, median, plates = inputs()
    stats.to_parquet(DATA / "ccd_completeness_reference.parquet", index=False)
    opp = opportunities(cat, prop, ref, plates)
    sample = sample_tables(cat, stats, ref, opp)
    footprint_check(sample, opp)
    cut_recovery_stages(sample, opp)
    correction_components(median)
    print(pd.read_csv(DATA / "plate_completeness_cut_sweep.csv").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--property-control", action="store_true")
    args = parser.parse_args()
    if args.property_control:
        property_control()
    else:
        main()
