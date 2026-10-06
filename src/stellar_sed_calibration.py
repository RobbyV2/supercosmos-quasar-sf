import argparse
import json
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from dust_extinction.parameter_averages import F99
from scipy.spatial import cKDTree

import assemble_lightcurves as al

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "temp/calibration_sed_model"
AB_OFFSET = np.array([0., 0., .015, .016, .035])
AB_ERROR = np.array([.019, .014, .004, .004, .011])
ASINH_B = np.array([1.4, .9, 1.2, 1.8, 7.4]) * 1e-10
BANDS = dict(zip("ugriz", [f"SLOAN_SDSS.{b}.dat" for b in "ugriz"]))
BANDS.update(bj="serc-j.txt", r_native="serc-r.txt", e="possi-e.txt", tp="techpan_og590.txt")
EBV = np.arange(0., .551, .005)
PASSBAND_URLS = dict(
    parker="https://www.cambridge.org/core/services/aop-cambridge-core/content/view/878CF0C11CE7BC44BCA9294637F46981/S132335800000597Xa.pdf/the-introduction-of-tech-pan-film-at-the-uk-schmidt-telescope.pdf",
    schott="https://www.schott.com/-/media/project/onex/products/o/optical-filter-glass/downloads/schott-datasheet-collection-filter-en_112019.pdf?rev=766d730fb2ac41fd82165f3f74ee7ae3")
PARKER_X = (133.2550048828125, 499.0790100097656)
PASSBANDS = dict(  # Parker & Malin 1999 Fig. 1 curve drawing, line items, frame drawing/item, frame bottom/top, printed nodes
    techpan=(7, slice(24, 50), (0, 0), 254.3599853515625, 98.00897216796875, False,
             "Derived 4415+OG590 3mm response: Parker & Malin 1999 Fig. 1 times SCHOTT transmission.\n"
             "wavelength_A relative_photon_response\n"
             "Energy sensitivity divided by wavelength for photon-counting integration; normalized peak=1.\n"
             "Not a measured total telescope/atmosphere response."),
    iiiaf=(14, slice(43, 71), (7, 50), 449.6199951171875, 293.447021484375, True,
           "Derived IIIaF+OG590 3mm: Parker & Malin 1999 Fig. 1 lower panel times SCHOTT filter table.\n"
           "wavelength_A relative_photon_response\n"
           "Energy sensitivity divided by wavelength; normalized peak=1. Not measured full telescope/atmosphere throughput."))


def redden(wave: np.ndarray, flux: np.ndarray, ebv: float) -> np.ndarray:
    return flux * F99(Rv=3.1).extinguish(np.asarray(wave) * u.AA, Ebv=ebv)


def catalog_to_ab(mag: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arg = -mag * np.log(10.) / 2.5 - np.log(ASINH_B)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        flux = 2. * ASINH_B * np.sinh(arg)
        return -2.5 * np.log10(flux) + AB_OFFSET, error / np.tanh(arg)


def star_inputs() -> pd.DataFrame:
    path = WORK / "star_inputs.parquet"
    if path.exists():
        saved = pd.read_parquet(path)
        if "u_ab_err" in saved:
            return saved
    pos = al.star_sky_positions()
    cat = pd.read_csv(DATA / "stripe82calibStars_v4.2.dat", sep=r"\s+", comment="#",
                      header=None, usecols=[1, 2, 6] + list(range(7, 37)))
    sky = SkyCoord(ra=pos[:, 0] * u.deg, dec=pos[:, 1] * u.deg)
    idx, sep, _ = sky.match_to_catalog_sky(SkyCoord(ra=cat[1].to_numpy() * u.deg,
                                                   dec=cat[2].to_numpy() * u.deg))
    cat = cat.iloc[idx].reset_index(drop=True)
    out = pd.DataFrame(dict(star=np.arange(len(pos)), ra=pos[:, 0], dec=pos[:, 1],
                            match_arcsec=sep.arcsec, ebv_sfd=cat[6] / 2.751))
    out["ebv"] = .86 * out.ebv_sfd
    good = (sep.arcsec <= 1.) & out.ebv.between(0., EBV[-1]).to_numpy()
    for i, band in enumerate("ugriz"):
        start = 7 + 6 * i
        out[band] = cat[start + 1]
        out[f"{band}_err"] = 1.25 * cat[start + 3]
        out[f"{band}_n"] = cat[start]
        good &= out[band].between(10., 30.).to_numpy() & (out[f"{band}_err"] > 0.)
        good &= cat[start] >= 4
        if band in "gri":
            good &= cat[start + 3] * np.sqrt(cat[start]) < .03
    ab, ab_error = catalog_to_ab(out[list("ugriz")].to_numpy(),
                                 out[[b + "_err" for b in "ugriz"]].to_numpy())
    for i, band in enumerate("ugriz"):
        out[f"{band}_ab"], out[f"{band}_ab_err"] = ab[:, i], ab_error[:, i]
    out["input_ok"] = good & np.all(np.isfinite(ab) & (ab_error > 0), axis=1)
    out.to_parquet(path, index=False)
    return out


def _weights(wave: np.ndarray, wf: np.ndarray, tf: np.ndarray) -> np.ndarray:
    lam = np.linspace(wf.min(), wf.max(), 8001)
    trans = np.interp(lam, wf, tf)
    step = (lam[-1] - lam[0]) / (len(lam) - 1)
    weights = trans * lam * step
    weights[[0, -1]] *= .5
    weights /= al._F0_AB * al._C_AA * np.trapezoid(trans / lam, lam)
    ix = np.searchsorted(wave, lam) - 1
    frac = (lam - wave[ix]) / (wave[ix + 1] - wave[ix])
    return (np.bincount(ix, weights * (1. - frac), minlength=len(wave))
            + np.bincount(ix + 1, weights * frac, minlength=len(wave)))


def model_grid(energy: bool = False, tilt: float = 0., og590: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = DATA / "bosz/synthetic_photometry.npz"
    key = "magnitudes_energy" if energy else "magnitudes"
    vkey = "vega_energy" if energy else "vega"
    if tilt:
        key, vkey = f"magnitudes_tilt{tilt:g}", f"vega_tilt{tilt:g}"
    if og590:
        key, vkey = "magnitudes_og590", "vega_og590"
    if path.exists():
        d = np.load(path)
        if key in d:
            return d["params"], d[key], d[vkey]
    manifest = json.loads((DATA / "bosz/manifest.json").read_text())["models"]
    wave = np.loadtxt(DATA / "bosz/bosz2024_wave_r500.txt")
    keep = (wave > 2500.) & (wave < 12000.)
    wave = wave[keep]
    flux = np.array([np.loadtxt(DATA / "bosz" / r["file"])[keep, 0] for r in manifest])
    filters = [al.read_bandpass(f) for f in BANDS.values()]
    if og590:
        filters[6] = al.read_bandpass("iiiaf_og590.txt")
    elif energy:
        filters = [(w, t / w) if 5 <= j <= 7 else (w, t) for j, (w, t) in enumerate(filters)]
    elif tilt:
        filters = [(w, t * (w / np.sqrt(w.min() * w.max())) ** tilt) if j >= 5 else (w, t)
                   for j, (w, t) in enumerate(filters)]
    weights = np.stack([_weights(wave, *f) for f in filters])
    vega = np.array([0.] * 5 + [al.ab_minus_vega(*f) for f in filters[5:]])
    ext = np.array([F99(Rv=3.1).extinguish(wave * u.AA, Ebv=e) for e in EBV])
    integrals = flux @ (ext[:, None, :] * weights[None, :, :]).reshape(-1, len(wave)).T
    mags = -2.5 * np.log10(integrals.reshape(len(flux), len(EBV), len(filters))) - vega
    params = np.array([[r["teff"], r["logg"], r["metallicity"]] for r in manifest])
    saved = dict(np.load(path)) if path.exists() else {}
    saved.update(params=params, ebv=EBV, bands=list(BANDS), ab_offset=AB_OFFSET, ab_error=AB_ERROR)
    saved.update({key: mags, vkey: vega})
    np.savez_compressed(path, **saved)
    return params, mags, vega


def dense_grid(params: np.ndarray, mags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, m = [], []
    for gravity, metal in np.unique(params[:, 1:], axis=0):
        idx = np.flatnonzero((params[:, 1] == gravity) & (params[:, 2] == metal))
        idx = idx[np.argsort(params[idx, 0])]
        temps = params[idx, 0]
        target = np.arange(temps[0], temps[-1] + 1., 25.)
        right = np.clip(np.searchsorted(temps, target, side="right"), 1, len(temps) - 1)
        frac = (target - temps[right - 1]) / (temps[right] - temps[right - 1])
        p.append(np.column_stack([target, np.full(len(target), gravity), np.full(len(target), metal)]))
        m.append(mags[idx[right - 1]] * (1 - frac[:, None, None])
                 + mags[idx[right]] * frac[:, None, None])
    return np.concatenate(p), np.concatenate(m).astype(np.float32)


def _at_ebv(mags: np.ndarray, ebv: np.ndarray, idx: np.ndarray) -> np.ndarray:
    lo = np.minimum((ebv / .005).astype(int), len(EBV) - 2)
    frac = (ebv - EBV[lo]) / .005
    return mags[idx, lo[:, None]] * (1. - frac[:, None, None]) + mags[idx, lo[:, None] + 1] * frac[:, None, None]


def fit_stars(obs: np.ndarray, err: np.ndarray, ebv: np.ndarray, params: np.ndarray,
              mags: np.ndarray, candidates: int = 128) -> dict[str, np.ndarray]:
    count = len(obs)
    result = dict(chi2=np.full(count, np.inf), model=np.zeros(count, int),
                  delta=np.zeros((count, 4)), delta_err=np.zeros((count, 4)),
                  residual=np.zeros((count, 5)))
    colors = obs[:, :-1] - obs[:, 1:]
    scale = np.array([.04, .02, .02, .025])
    group = np.rint(ebv / .01).astype(int)
    for g in np.unique(group):
        inds = np.flatnonzero(group == g)
        grid = mags[:, np.clip(g * 2, 0, len(EBV) - 1), :5]
        gc = grid[:, :-1] - grid[:, 1:]
        tree = cKDTree(gc / scale)
        gri = cKDTree(gc[:, 1:] / scale[1:])
        for start in range(0, len(inds), 2048):
            ix = inds[start:start + 2048]
            full = tree.query(colors[ix] / scale, k=candidates, workers=1)[1]
            red = gri.query(colors[ix, 1:] / scale[1:], k=candidates // 2, workers=1)[1]
            cand = np.sort(np.concatenate([full, red], axis=1), axis=1)
            duplicate = np.column_stack([np.zeros(len(ix), bool), np.diff(cand, axis=1) == 0])
            pred = _at_ebv(mags, ebv[ix], cand)
            residual = obs[ix, None, :] - pred[:, :, :5]
            weight = 1. / err[ix] ** 2
            norm = np.sum(residual * weight[:, None, :], axis=2) / weight.sum(axis=1)[:, None]
            residual -= norm[:, :, None]
            chi = np.sum(residual ** 2 * weight[:, None, :], axis=2)
            chi[duplicate] = np.inf
            best = np.argmin(chi, axis=1)
            minimum = chi[np.arange(len(ix)), best]
            prob = np.exp(-.5 * np.minimum(chi - minimum[:, None], 150.))
            prob /= prob.sum(axis=1)[:, None]
            delta = pred[:, :, [1, 2, 2, 2]] - pred[:, :, 5:] - AB_OFFSET[[1, 2, 2, 2]]
            mean = np.sum(delta * prob[:, :, None], axis=1)
            result["chi2"][ix] = minimum
            result["model"][ix] = cand[np.arange(len(ix)), best]
            result["delta"][ix] = delta[np.arange(len(ix)), best]
            result["delta_err"][ix] = np.sqrt(np.sum((delta - mean[:, None, :]) ** 2 * prob[:, :, None], axis=1))
            result["residual"][ix] = residual[np.arange(len(ix)), best]
    return result


def run() -> None:
    WORK.mkdir(exist_ok=True, parents=True)
    table = star_inputs()
    params, mags, _ = model_grid()
    params, mags = dense_grid(params, mags)
    ok = table.input_ok.to_numpy()
    idx = np.flatnonzero(ok)
    obs = table.loc[ok, [f"{b}_ab" for b in "ugriz"]].to_numpy()
    errors = table.loc[ok, [f"{b}_ab_err" for b in "ugriz"]].to_numpy()
    errors = np.hypot(errors, AB_ERROR)
    ebv = table.ebv.to_numpy()[ok]
    print(f"Fitting {len(idx)} stars against {len(params)} interpolated spectra", flush=True)
    fixed = fit_stars(obs, errors, ebv, params, mags)
    correction = (obs - AB_OFFSET - table.loc[ok, list("ugriz")].to_numpy())[:, [1, 2, 2, 2]]
    for j, name in enumerate(["teff", "logg", "metallicity"]):
        table.loc[ok, name] = params[fixed["model"], j]
    table.loc[ok, "chi2"] = fixed["chi2"]
    table["dof"] = 1
    table["grid_edge"] = False
    table.loc[ok, "grid_edge"] = np.any((params[fixed["model"]] == params.min(axis=0))
                                         | (params[fixed["model"]] == params.max(axis=0)), axis=1)
    for j, band in enumerate("ugriz"):
        table.loc[ok, f"{band}_residual"] = fixed["residual"][:, j]
    native_names = ["bj", "r", "e", "tp"]
    delta_names = ["delta_g_bj", "delta_r_r", "delta_r_e", "delta_r_tp"]
    for j, (name, delta, ref) in enumerate(zip(native_names, delta_names, ["g", "r", "r", "r"])):
        table.loc[ok, delta] = fixed["delta"][:, j] - correction[:, j]
        table.loc[ok, f"{delta}_err"] = fixed["delta_err"][:, j]
        table[f"{name}_native"] = table[ref] - table[delta]
        table[f"{name}_native_err"] = np.hypot(table[f"{ref}_err"], table[f"{delta}_err"])
    table["fit_ok"] = ok & (table.chi2 < 25.)
    table.loc[ok, "fit_ok"] &= np.max(np.abs(fixed["residual"]), axis=1) < .1
    _save(table)
    print(f"Fixed-column fits saved: {table.fit_ok.sum()} accepted", flush=True)
    best = {k: v.copy() for k, v in fixed.items()}
    best_ebv = ebv.copy()
    for fraction in [0., .25, .5, .75]:
        alternative = fit_stars(obs, errors, fraction * ebv, params, mags)
        win = alternative["chi2"] < best["chi2"]
        for k in best:
            best[k][win] = alternative[k][win]
        best_ebv[win] = fraction * ebv[win]
        print(f"Reddening fraction {fraction:g}: {win.sum()} lower chi-squared", flush=True)
    table.loc[ok, "ebv_bounded"] = best_ebv
    table.loc[ok, "chi2_bounded"] = best["chi2"]
    for j, name in enumerate(native_names):
        table.loc[ok, f"{name}_native_bounded_shift"] = fixed["delta"][:, j] - best["delta"][:, j]
    _save(table)
    qualified, good = table[table.input_ok], table[table.fit_ok]
    summary = dict(counts=dict(all=len(table), input_ok=len(qualified), fit_ok=len(good),
                              chi2_ge25=int((qualified.chi2 >= 25).sum()),
                              max_resid_ge01=int((qualified[[b + "_residual" for b in "ugriz"]].abs().max(axis=1) >= .1).sum()),
                              grid_edge_fit_ok=int(good.grid_edge.sum())),
                   n_grid_models=len(params),
                   ranges={c: good[c].quantile([.01, .5, .99]).tolist()
                           for c in ["g", "r", "ebv", "teff", "logg", "metallicity"]},
                   errors={b: good[b + "_native_err"].quantile([.5, .95, .99]).tolist() for b in native_names},
                   foreground={b: good[b + "_native_bounded_shift"].abs().quantile([.5, .95, .99]).tolist()
                               for b in native_names},
                   color_g_r_quantiles=(good.g - good.r).quantile([.01, .5, .99]).tolist(),
                   quality_chi2_median=float(good.chi2.median()),
                   bounded_reddening_fraction_lt_half=float((good.ebv_bounded < .5 * good.ebv).mean()),
                   bounded_chi2_improvement_gt9=float(((good.chi2 - good.chi2_bounded) > 9).mean()))
    (DATA / "bosz/fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary, flush=True)


def _save(table: pd.DataFrame) -> None:
    path = DATA / "standard_star_sed.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def native_response_sensitivity(tilts: bool = False, og590: bool = False) -> None:
    base_params, primary, _ = model_grid()
    params, primary = dense_grid(base_params, primary)
    table = pd.read_parquet(DATA / "standard_star_sed.parquet")
    ok = table.input_ok.to_numpy()
    keys = pd.MultiIndex.from_arrays(params.T)
    indices = keys.get_indexer(pd.MultiIndex.from_frame(table.loc[ok, ["teff", "logg", "metallicity"]]))
    if np.any(indices < 0):
        raise ValueError("Fitted atmosphere missing from response sensitivity grid")
    response_path = DATA / "standard_star_sed_response.parquet"
    out = (pd.read_parquet(response_path) if response_path.exists() else table[["star"]].copy()) if tilts or og590 else table
    ebv = table.loc[ok, "ebv"].to_numpy()
    if og590:
        _, alternate, _ = model_grid(og590=True)
        _, alternate = dense_grid(base_params, alternate)
        obs = table.loc[ok, [f"{b}_ab" for b in "ugriz"]].to_numpy()
        errors = np.hypot(table.loc[ok, [f"{b}_ab_err" for b in "ugriz"]].to_numpy(), AB_ERROR)
        result = fit_stars(obs, errors, ebv, params, alternate)
        np.testing.assert_array_equal(result["model"], indices)
        np.testing.assert_allclose(result["chi2"], table.loc[ok, "chi2"], atol=1e-10)
        correction = obs[:, 2] - AB_OFFSET[2] - table.loc[ok, "r"].to_numpy()
        native = table.loc[ok, "r"].to_numpy() - result["delta"][:, 1] + correction
        out.loc[ok, "r_native_og590_shift"] = native - table.loc[ok, "r_native"].to_numpy()
        out.loc[ok, "r_native_og590_err"] = np.hypot(table.loc[ok, "r_err"], result["delta_err"][:, 1])
    bands = ["bj", "r", "e", "tp"] if tilts else ["bj", "r", "e"]
    for tilt in (() if og590 else ((1, 2) if tilts else (0,))):
        _, alternate, _ = model_grid(tilt=tilt, energy=not tilts)
        _, alternate = dense_grid(base_params, alternate)
        shift = np.zeros((len(indices), len(bands)))
        for start in range(0, len(indices), 8192):
            sl = slice(start, start + 8192)
            args = ebv[sl], indices[sl, None]
            shift[sl] = (_at_ebv(alternate, *args) - _at_ebv(primary, *args))[:, 0, 5:5 + len(bands)]
        label = f"tilt{tilt}" if tilts else "energy"
        for j, band in enumerate(bands):
            out.loc[ok, f"{band}_native_{label}_shift"] = shift[:, j]
    if tilts or og590:
        temporary = response_path.with_suffix(".parquet.tmp")
        out.to_parquet(temporary, index=False)
        temporary.replace(response_path)
    else:
        _save(out)
    print(out.filter(regex="native_.*_shift$").quantile([.01, .5, .99]).to_string(), flush=True)


def build_passbands(out: Path = DATA) -> None:
    import urllib.request
    import pymupdf

    src = ROOT / "temp/passband_sources"
    src.mkdir(parents=True, exist_ok=True)
    pdf = {}
    for name, url in PASSBAND_URLS.items():
        if not (src / f"{name}.pdf").exists():
            urllib.request.urlretrieve(url, src / f"{name}.pdf")
        pdf[name] = pymupdf.open(src / f"{name}.pdf")
    words = pdf["schott"][91].get_text("words")
    schott = []
    for x0, y0, _, _, word, *_ in words:
        if 140 < x0 < 150 and word.isdigit() and 500 <= int(word) <= 790:
            value, = [a[4] for a in words if 170 < a[0] < 180 and abs(a[1] - y0) < .02]
            schott.append((float(word) * 10, 0. if int(word) < 550 else float(value.replace(",", "."))))
    schott = np.array(schott)
    drawings = pdf["parker"][1].get_drawings()
    for name, (draw, items, (fd, fi), bottom, top, printed, header) in PASSBANDS.items():
        frame = drawings[fd]["items"][fi][1].rect
        seg = drawings[draw]["items"][items]
        x, y = np.array([seg[0][1]] + [z[2] for z in seg]).T
        if (len(schott) != 30 or len(seg) != items.stop - items.start or any(z[0] != "l" for z in seg)
                or np.any(np.diff(x) <= 0) or not np.allclose([frame.x0, frame.x1, frame.y1, frame.y0],
                                                              [*PARKER_X, bottom, top], atol=.001)):
            raise ValueError(f"Parker & Malin or SCHOTT layout changed ({name})")
        wave, sens = 4500 + (x - PARKER_X[0]) * 3000 / (PARKER_X[1] - PARKER_X[0]), (bottom - y) / (bottom - top)
        if printed:
            wave, sens = np.char.mod("%.6f", wave).astype(float), np.char.mod("%.8f", sens).astype(float)
        grid = np.unique(np.r_[np.arange(5000., 7500.1, 10), wave[(wave >= 5000) & (wave <= 7500)],
                               schott[:, 0][schott[:, 0] <= 7500]])
        photon = np.interp(grid, wave, sens) * np.interp(grid, *schott.T, left=0, right=0) / grid
        np.savetxt(out / f"{name}_og590.txt", np.c_[grid, photon / photon.max()], fmt=["%.6f", "%.9f"], header=header)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-sensitivity", action="store_true")
    parser.add_argument("--response-tilts", action="store_true")
    parser.add_argument("--og590", action="store_true")
    parser.add_argument("--build-passbands", nargs="?", const=DATA, type=Path)
    args = parser.parse_args()
    if args.build_passbands:
        build_passbands(args.build_passbands)
    elif args.og590:
        native_response_sensitivity(og590=True)
    elif args.response_tilts:
        native_response_sensitivity(tilts=True)
    elif args.response_sensitivity:
        native_response_sensitivity()
    else:
        run()
