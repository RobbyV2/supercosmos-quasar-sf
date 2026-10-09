import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_structure_function as esf

_ROOT = Path(__file__).resolve().parents[1]

CCD_MJD = 53200.0  # mean epoch, 1998-2007 SDSS window
NULL_LAG = np.logspace(1.0, 4.5, 11)


def _fiducial_limits() -> pd.DataFrame:
    from plate_completeness import BRIGHT_BOUND
    s = pd.read_parquet(_ROOT / "data/plate_completeness_sample.parquet")
    return s[s.adopted_keep].groupby("band").reference_mag.agg(["min", "max"]).assign(min=BRIGHT_BOUND)


def main() -> None:
    import plate_magnitude_models as native
    native.native_variant("effective", set())
    native.native_null()


if __name__ == "__main__":
    esf.check_flags(__file__)
    main()
