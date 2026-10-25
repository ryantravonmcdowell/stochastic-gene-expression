"""Statistical checks that take minutes; run with ``pytest -m slow``."""

import pytest
from scipy import stats

from burstkin.validate import coverage_study


@pytest.mark.slow
def test_profile_intervals_have_nominal_coverage():
    replicates = 200
    coverage = coverage_study(replicates=replicates)
    # 99.9% binomial acceptance region around the nominal 95%
    lo, hi = stats.binom.interval(0.999, replicates, 0.95)
    for target, value in coverage.items():
        assert lo / replicates <= value <= hi / replicates, (target, value)
