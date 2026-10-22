# Quasar ensemble structure functions over a 72-year baseline

SuperCOSMOS plates + SDSS/PS1/ZTF, Stripe 82.

## Setup

To create the environment:

```sh
python3.12 -m venv .venv && py=.venv/bin/python && $py -m pip install -r requirements.txt
```

## Data

Included:

- `data/`: SDSS, SERC, POSS-I passbands; CALSPEC Vega; Vanden Berk composite; Roma-BZCAT; BOSZ and SDSS spectrum manifests; SuperCOSMOS `wu_qso.hdf5`, `survey_table.fits`; IRSA ZTF exposure metadata (`ztf.ztf_current_meta_sci`)
- `data/plate_quasar_epochs.csv`: 25537 calibrated plate epochs, SDSS g/r system, DR16Q quasars less Roma-BZCAT blazars; columns `sdss_name ra dec band mjd mag magerr plate survey fiducial_g fiducial_r`
- `data/plate_native_pair_variance.csv`: 91 signed residual calibration variances (mag^2) by native pair class and 0.5 mag bin, subtracted per pair by `floor_variance` in `src/ensemble_structure_function.py`; columns `native_a native_b band mag_lo mag_hi subset boundary_sigma_min n_pairs n_core n_stars class_id clip_center clip_limit residual_variance null_mean unclipped_null_mean null_star_mean training_floor_se null_star_se`

Not included:

- `data/S82/{Catalog,dr16s82_sdssLCRaw,dr16s82_ps1LCRaw,dr16s82_ZuberLCRaw}.parquet` (catalog, CCD light curves)
- `data/StandardStars.zip` (CCD standard stars), `unzip -d temp data/StandardStars.zip`

## Downloads

To retrieve the data:

```sh
# catalogs, dust maps, query notebook
curl -L http://quasar.astro.illinois.edu/paper_data/DR16Q/dr16q_prop_May01_2024.fits.gz | gunzip > data/dr16q_prop_May01_2024.fits
curl -Lo data/stripe82calibStars_v4.2.dat https://faculty.washington.edu/ivezic/sdss/calib82/dataV2/stripe82calibStars_v4.2.dat
curl --create-dirs -Lo data/S82/macleod2012_s82_lightcurves.tar.gz https://faculty.washington.edu/ivezic/macleod/qso_dr7/QSO_S82.tar.gz
for p in ngp sgp; do curl -L --create-dirs -o data/sfd/SFD_dust_4096_$p.fits https://raw.githubusercontent.com/kbarbary/sfddata/7a5fe7fadf086561ba4748756e59a4c51d0ec632/SFD_dust_4096_$p.fits; done
curl -Lo data/stone2022_ensemble_sf_psd.fits.gz https://zenodo.org/records/7624056/files/EnsDat.fits.gz
curl -Lo data/stone2022_lightcurves.fits.gz https://zenodo.org/records/7624056/files/TotalDat.fits.gz
curl --create-dirs -Lo src/query.ipynb https://raw.githubusercontent.com/burke86/supercosmos_qso/63893cfc4555cb3e09cba98e650e678fc6146a5a/query.ipynb
# BOSZ models and SDSS spectra, checksums verified
$py -c "import json; [print(m['url'], m['sha256'], m['file']) for m in json.load(open('data/bosz/manifest.json'))['models']]" | while read url sha f; do curl -sSfLo data/bosz/$f $url; echo "$sha  data/bosz/$f"; done | sha256sum -c --quiet
tail -n +2 data/sdss_sed_control/manifest.csv | while IFS=, read plate mjd fiber f url sha; do curl -sSfLo data/sdss_sed_control/$f $url; echo "$sha  data/sdss_sed_control/$f"; done | sha256sum -c --quiet
# standard-star plate detections, SuperCOSMOS Science Archive, about 3 days
$py -m pip install pyvo==1.9.1 tqdm==4.70.1
$py -c "import json, pandas as pd; exec(''.join(json.load(open('src/query.ipynb'))['cells'][8]['source'])); c = pd.read_csv('data/stripe82calibStars_v4.2.dat', sep=r'\s+', comment='#'); query_ssa_service(SkyCoord(c.iloc[:, 1].to_numpy() * u.deg, c.iloc[:, 2].to_numpy() * u.deg), output_file='data/roe_stars.hdf5')"
```

## Run

To run in order from this folder (writes `data/`, `temp/`):

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1  # single-thread BLAS; fixed summation order
set -eo pipefail  # stop at the first failure
$py src/stellar_sed_calibration.py --build-passbands  # IIIa-F, Tech Pan + OG590 passbands from public PDFs
$py src/stellar_sed_calibration.py; for m in --response-sensitivity --og590; do $py src/stellar_sed_calibration.py $m; done  # star SEDs, response checks
for m in --native-fit --native-quasars --native-noise --native-publish; do $py src/plate_magnitude_models.py $m --native-variant effective; done  # plate calibration
$py src/assemble_lightcurves.py  # light curves
$py src/plate_completeness.py  # completeness, samples
$py src/fit_structure_function.py --object-first  # fiducial SF
$py src/plate_magnitude_models.py --native-validation --native-variant effective  # calibration validation
$py src/sf_sampling_checks.py  # asymmetry, single pair
$py src/plate_completeness.py --property-control  # property control
$py src/star_null_structure_function.py  # star null
$py src/sf_model_checks.py; for m in --noise-fraction --fit-range --plate-replacement --lowmass-influence --band-ratio; do $py src/sf_model_checks.py $m; done  # DRW and bending-PSD fits, checks
$py src/plate_magnitude_models.py --native-floor-impact --native-variant effective  # residual-variance impact
$py src/fit_structure_function.py --literature-choices  # literature comparison
$py src/fit_structure_function.py --refit-object-first  # group SFs, rises
$py src/sf_model_checks.py --group-trends  # group trends
$py src/simulate_structure_function.py --aggregation-test; $py src/simulate_structure_function.py --excess-null  # DRW recovery, excess null
for m in --native-sed-control --observed-sed-control; do $py src/sss_qsogen_offsets.py $m; done  # quasar SED controls
$py src/plate_magnitude_models.py --native-sed-impact --native-variant effective  # SED sensitivity
```

## License

MIT ([LICENSE](LICENSE))
