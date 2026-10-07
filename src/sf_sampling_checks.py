from pathlib import Path
import hashlib
import inspect

import numpy as np
import pandas as pd

import ensemble_structure_function as esf
from plate_completeness import load_completeness_lightcurves

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("all", "ccd_ccd", "plate_ccd", "ztf_ztf")
SIGNS = ("total", "brightening", "dimming")
CACHE = esf.CACHE / "sampling"


def pair_values(sub, redshift, edges=esf.EDGES):
    sub = sub.sort_values("night", kind="stable")
    t, mag, sig, vt = (sub[c].to_numpy(float) for c in ("night", "mag", "sig", "trunc_v"))
    i, j = np.triu_indices(len(sub), 1)
    bins = np.searchsorted(edges, (t[j] - t[i]) / (1 + redshift), side="right") - 1
    good = (bins >= 0) & (bins < len(edges) - 1)
    i, j, bins = i[good], j[good], bins[good]
    dm = mag[j] - mag[i]
    w = 2 / (vt[i] + vt[j])
    noise = w * (sig[i] ** 2 + sig[j] ** 2
                 + sub[esf.FV_COLS].to_numpy(float)[i, sub.pcode.to_numpy()[j]])
    plate = sub.sss.to_numpy(bool)
    ztf = sub.survey.eq("ztf").to_numpy()
    kinds = (np.ones(len(i), bool), ~plate[i] & ~plate[j], plate[i] ^ plate[j], ztf[i] & ztf[j])
    return bins, dm, w * dm ** 2, noise, kinds, sub.plate.to_numpy()[i], sub.plate.to_numpy()[j]


def split_accumulator(bins, dm, dm2, noise, kinds, nb=esf.NB):
    out = np.zeros((len(kinds), 3, nb, 3))
    for k, keep in enumerate(kinds):
        for s, sign in enumerate((np.ones(len(dm), bool), dm < 0, dm > 0)):
            use = keep & sign
            out[k, s, :, 0] = np.bincount(bins[use], minlength=nb)
            out[k, s, :, 1] = np.bincount(bins[use], weights=dm2[use], minlength=nb)
            out[k, s, :, 2] = np.bincount(bins[use], weights=noise[use], minlength=nb)
    return out


def draw_single_pairs(bins, moment, keep, rng, n_draw, nb=esf.NB):
    out = np.full((n_draw, nb), np.nan)
    for k in np.unique(bins[keep]):
        values = moment[keep & (bins == k)]
        out[:, k] = values[rng.integers(0, len(values), n_draw)]
    return out


def amplitude(value):
    return np.sqrt(np.where(value > 0, value, np.nan))


def finite_mean(value, axis=0):
    count = np.isfinite(value).sum(axis=axis)
    total = np.nansum(value, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def asymmetry(moment):
    sf = amplitude(moment)
    return (sf[..., 1, :] - sf[..., 2, :]) / sf[..., 0, :]


def signature(ids, n_draw):
    paths = [ROOT / esf.LC_PARQUET, ROOT / "data/detection_correction_components.parquet",
             ROOT / "data/plate_native_pair_variance.csv", ROOT / "data/plate_completeness_sample.parquet"]
    source = "".join(inspect.getsource(f) for f in (pair_values, split_accumulator, draw_single_pairs,
                                                    *esf.KERNELS, load_completeness_lightcurves))
    stamp = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(ids.astype("U").tobytes() + repr((stamp, n_draw)).encode() + source.encode()).hexdigest()


def build_cache(sample, n_draw):
    ids = np.sort(sample.OBJID.unique()).astype(str)
    zmap = sample.drop_duplicates("OBJID").set_index("OBJID").z
    selected = sample[sample.adopted_keep].set_index(["OBJID", "band"]).index
    usable = sample.loc[np.isfinite(sample.reference_mag), "OBJID"].unique()
    token = signature(ids, n_draw)
    CACHE.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(ids), 1000):
        part = ids[start:start + 1000]
        path = CACHE / f"chunk_{start:05d}.npz"
        if path.exists():
            with np.load(path) as f:
                if str(f["signature"]) == token:
                    continue
        nt = esf.clean_nightly(load_completeness_lightcurves(part[np.isin(part, usable)]))[0]
        index = {o: i for i, o in enumerate(part)}
        acc = np.zeros((len(part), 2, len(KINDS), 3, esf.NB, 3))
        draws, draw_keys, deletions, delete_keys = [], [], [], []
        rng = np.random.default_rng(928000 + start)
        for (oid, band), sub in nt.groupby(["OBJID", "band"], sort=False, observed=True):
            bi, qi = esf.BANDS.index(band), index[oid]
            values = pair_values(sub, zmap[oid])
            bins, dm, dm2, noise, kinds, plate_i, plate_j = values
            acc[qi, bi] = split_accumulator(bins, dm, dm2, noise, kinds)
            if (oid, band) in selected:
                draws.append(draw_single_pairs(bins, dm2 - noise, kinds[2], rng, n_draw))
                draw_keys.append((oid, band))
                plates = np.unique(np.concatenate([plate_i[plate_i >= 0], plate_j[plate_j >= 0]]))
                for plate in plates:
                    touched = (plate_i == plate) | (plate_j == plate)
                    deletions.append(split_accumulator(bins, dm, dm2, noise,
                                                       [k & touched for k in kinds]))
                    delete_keys.append((oid, band, str(plate)))
        np.savez_compressed(path, signature=token, ids=part, acc=acc,
                            draws=np.array(draws), draw_keys=np.array(draw_keys, dtype=str),
                            deletions=np.array(deletions), delete_keys=np.array(delete_keys, dtype=str))
        print(f"Accumulated {min(start + 1000, len(ids))}/{len(ids)} quasars", flush=True)
    chunks = [np.load(CACHE / f"chunk_{start:05d}.npz") for start in range(0, len(ids), 1000)]
    out = {key: np.concatenate([f[key] for f in chunks if f[key].size], axis=0)
           for key in ("ids", "acc", "draws", "draw_keys", "deletions", "delete_keys")}
    for f in chunks:
        f.close()
    return out


def joint_test(point, reps, keep):
    good = keep & np.isfinite(point) & np.all(np.isfinite(reps), axis=0)
    x, boot = point[good], reps[:, good]
    if not len(x):
        return dict(n_bins=0, max_abs_z=np.nan, global_p_bootstrap=np.nan)
    sigma = boot.std(axis=0, ddof=1)
    valid = sigma > 0
    x, boot, sigma = x[valid], boot[:, valid], sigma[valid]
    observed = np.max(np.abs(x / sigma))
    null = np.max(np.abs((boot - boot.mean(axis=0)) / sigma), axis=1)
    return dict(n_bins=len(x), max_abs_z=observed,
                global_p_bootstrap=(1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def paired_single_bootstrap(draws, acc, n_boot, seed):
    rng = np.random.default_rng(seed)
    moments = esf.object_second_moments(acc)
    out = np.full((n_boot, 2, acc.shape[-2]), np.nan)
    for b in range(n_boot):
        ids = rng.integers(0, len(acc), len(acc))
        selection = rng.integers(0, draws.shape[1], len(acc))
        out[b, 0] = finite_mean(moments[ids])
        out[b, 1] = finite_mean(draws[ids, selection])
    return out


def plate_delete(acc, ids, band, cache):
    keys = cache["delete_keys"]
    valid = (keys[:, 1] == band) & np.isin(keys[:, 0], ids)
    keys, removed = keys[valid], cache["deletions"][valid]
    plates = np.unique(keys[:, 2])
    positions = {o: i for i, o in enumerate(ids)}
    point = esf.object_second_moments(acc)
    numer = np.nansum(point, axis=0)
    denom = np.isfinite(point).sum(axis=0)
    estimates = []
    for plate in plates:
        use = keys[:, 2] == plate
        idx = np.array([positions[o] for o in keys[use, 0]])
        left = acc[idx] - removed[use]
        rem = esf.object_second_moments(left)
        num = numer + (np.nan_to_num(rem) - np.nan_to_num(point[idx])).sum(axis=0)
        den = denom + (np.isfinite(rem).astype(int) - np.isfinite(point[idx]).astype(int)).sum(axis=0)
        estimates.append(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))
    return plates, np.array(estimates)


def analyse(cache, sample, n_boot):
    ids, acc = cache["ids"], cache["acc"]
    summaries, tests, single_rows, delete_rows, memberships = [], [], [], [], []
    covariance = dict(lag_edges=esf.EDGES, pair_classes=np.array(KINDS), signs=np.array(SIGNS),
                      n_boot=np.array(n_boot))
    baseline = pd.read_csv(ROOT / "data/structure_function_fixed_population.csv")
    for bi, band in enumerate(esf.BANDS):
        s = sample[sample.band.eq(band)].set_index("OBJID").reindex(ids)
        endpoint = 12 if band == "g" else 13
        selected = s.adopted_keep.fillna(False).to_numpy(bool)
        fiducial = selected & (acc[:, bi, 0, 0, :endpoint + 1, 0] > 0).all(axis=1)
        parent = (acc[:, bi, 0, 0, :, 0] > 0).any(axis=1)
        memberships.extend(dict(OBJID=o, band=band, full_parent=bool(p), selected=bool(v), fiducial=bool(f))
                           for o, p, v, f in zip(ids, parent, selected, fiducial))
        for name, mask in (("full_parent", parent), ("selected", selected), ("fiducial", fiducial)):
            a, oid = acc[mask, bi], ids[mask]
            point = esf.ensemble_second_moment(a)
            bounds, reps = esf.object_bootstrap(a, n_boot=n_boot, seed=928 + bi)
            beta, breps = asymmetry(point), asymmetry(reps)
            covariance[f"{name}_{band}_second_moment"] = np.cov(reps.reshape(n_boot, -1), rowvar=False)
            covariance[f"{name}_{band}_beta"] = np.cov(breps.reshape(n_boot, -1), rowvar=False)
            covariance[f"{name}_{band}_beta_replicates"] = breps
            covariance[f"{name}_{band}_second_moment_replicates"] = reps
            for ki, kind in enumerate(KINDS):
                nobj = (a[:, ki, :, :, 0] > 0).sum(axis=0)
                ncommon = (a[:, ki, 1:, :, 0] > 0).all(axis=1).sum(axis=0)
                count = a[:, ki, :, :, 0].sum(axis=0)
                common = (a[:, ki, 1:, :, 0] > 0).all(axis=1)
                common_point = esf.ensemble_second_moment(a[:, ki], weights=common[:, None, :])
                for k in range(esf.NB):
                    valid = ((nobj[1, k] >= 20) & (nobj[2, k] >= 20) & np.isfinite(beta[ki, k])
                             & (k <= endpoint if name == "fiducial" else True))
                    blo, bmid, bhi = (np.nanpercentile(breps[:, ki, k], [16, 50, 84])
                                     if np.isfinite(breps[:, ki, k]).any() else [np.nan] * 3)
                    row = dict(sample=name, band=band, pair_class=kind, bin=k,
                               lag_center_days=esf.CENTERS[k], lag_lo_days=esf.EDGES[k], lag_hi_days=esf.EDGES[k+1],
                               n_sample=len(oid), n_common_sign_objects=int(ncommon[k]), reportable=bool(valid),
                               beta=beta[ki, k], beta_lo=blo, beta_hi=bhi,
                               beta_common_objects=asymmetry(common_point)[k])
                    for j, sign in enumerate(SIGNS):
                        row.update({f"{sign}_n_objects":int(nobj[j, k]), f"{sign}_n_pairs":int(count[j, k]),
                                    f"{sign}_sf2_mag2":point[ki, j, k], f"{sign}_sf_mag":amplitude(point)[ki, j, k],
                                    f"{sign}_sf_lo":amplitude(bounds)[0, ki, j, k],
                                    f"{sign}_sf_hi":amplitude(bounds)[2, ki, j, k]})
                    summaries.append(row)
                for region, use in (("all_supported", np.ones(esf.NB, bool)),
                                    ("ccd_timescales", esf.CENTERS <= 1334),
                                    ("long_timescales", esf.CENTERS >= 2371)):
                    support = (nobj[1] >= 20) & (nobj[2] >= 20)
                    if name == "fiducial":
                        support[endpoint + 1:] = False
                    tests.append(dict(sample=name, band=band, pair_class=kind, region=region,
                                      **joint_test(beta[ki], breps[:, ki], support & use)))
            if name == "fiducial":
                covariance[f"fiducial_ids_{band}"] = oid
                covariance[f"fiducial_acc_{band}"] = a[:, 0, 0, :endpoint + 1]
                covariance[f"fiducial_asymmetry_acc_{band}"] = a
                expected = baseline[(baseline.band == band) & (baseline["sample"] == "fixed")]
                np.testing.assert_allclose(amplitude(point[0, 0, :endpoint + 1]), expected.sf_mag, rtol=1e-10)
                if len(oid) != int(expected.n_objects.iloc[0]):
                    raise AssertionError("fiducial membership differs from adopted sample")
                plates, deleted = plate_delete(a, oid, band, cache)
                covariance[f"fiducial_{band}_plate_ids"] = plates
                covariance[f"fiducial_{band}_plate_delete_second_moment"] = deleted
                for ip, plate in enumerate(plates):
                    for ki in (0, 2):
                        for k in range(endpoint + 1):
                            delete_rows.append(dict(band=band, plate=int(plate), pair_class=KINDS[ki], bin=k,
                                lag_center_days=esf.CENTERS[k], beta=asymmetry(deleted[ip])[ki, k],
                                sf_mag=amplitude(deleted[ip])[ki, 0, k]))
            if name != "full_parent":
                keys = cache["draw_keys"]
                take = (keys[:, 1] == band) & np.isin(keys[:, 0], oid)
                positions = {o: i for i, o in enumerate(oid)}
                order = np.array([positions[o] for o in keys[take, 0]])
                one = cache["draws"][take]
                single_mean = finite_mean(one)
                all_point = esf.ensemble_second_moment(a[order, 2, 0])
                covariance[f"{name}_{band}_single_pair_selection_moments"] = single_mean
                paired = paired_single_bootstrap(one, a[order, 2, 0], n_boot, 1928 + bi)
                covariance[f"{name}_{band}_single_pair_joint_replicates"] = paired
                for k in range(esf.NB):
                    n = np.isfinite(one[:, 0, k]).sum()
                    if n < 20 or (name == "fiducial" and k > endpoint):
                        continue
                    curve = amplitude(single_mean[:, k])
                    low, median, high = np.nanpercentile(curve, [16, 50, 84])
                    primary = amplitude(point)[2, 0, k]
                    err = np.nanstd(amplitude(reps[:, 2, 0, k]), ddof=1)
                    boot_single = amplitude(paired[:, 1, k])
                    boot_difference = boot_single - amplitude(paired[:, 0, k])
                    single_rows.append(dict(sample=name, band=band, bin=k, lag_center_days=esf.CENTERS[k],
                        n_objects=int(n), n_draw=one.shape[1], all_pairs_sf2_mag2=all_point[k],
                        all_pairs_sf_mag=primary, all_pairs_boot_err_mag=err,
                        all_pairs_sf_lo=amplitude(bounds)[0, 2, 0, k], all_pairs_sf_hi=amplitude(bounds)[2, 2, 0, k],
                        single_pair_mean_sf2_mag2=np.nanmean(single_mean[:, k]),
                        single_pair_mean_moment_sf_mag=amplitude(np.nanmean(single_mean[:, k])),
                        single_pair_sf_lo=low, single_pair_sf_median=median, single_pair_sf_hi=high,
                        single_pair_selection_sd_mag=np.nanstd(curve, ddof=1),
                        single_pair_object_and_selection_boot_sd_mag=np.nanstd(boot_single, ddof=1),
                        paired_difference_lo=np.nanpercentile(boot_difference, 16),
                        paired_difference_hi=np.nanpercentile(boot_difference, 84),
                        selection_sd_over_object_boot_err=np.nanstd(curve, ddof=1)/err))
            print(f"{name} {band}: {len(oid)} quasars", flush=True)
    pd.DataFrame(summaries).to_csv(ROOT / "data/structure_function_asymmetry.csv", index=False)
    pd.DataFrame(tests).to_csv(ROOT / "data/structure_function_asymmetry_tests.csv", index=False)
    pd.DataFrame(single_rows).to_csv(ROOT / "data/structure_function_single_pair.csv", index=False)
    pd.DataFrame(delete_rows).to_csv(ROOT / "data/structure_function_sampling_plate_deletion.csv", index=False)
    pd.DataFrame(memberships).to_parquet(ROOT / "data/structure_function_sampling_membership.parquet", index=False)
    np.savez_compressed(ROOT / "data/structure_function_sampling_covariance.npz", **covariance)


def main():
    esf.check_flags(__file__)
    sample = pd.read_parquet(ROOT / "data/plate_completeness_sample.parquet")
    sample["OBJID"] = sample.OBJID.astype(str)
    analyse(build_cache(sample, 1000), sample, 3000)


if __name__ == "__main__":
    main()
