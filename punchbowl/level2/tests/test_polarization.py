import pytest
from prefect.testing.utilities import prefect_test_harness
from prefect.logging import disable_run_logger

from punchbowl.data import PUNCHData
from punchbowl.tests.test_data import sample_wcs, sample_data_random, sample_punchdata, \
    sample_punchdata_list
from punchbowl.level2.polarization import define_amatrix, resolve_polarization, resolve_polarization_task

@pytest.mark.prefect_test
def test_resolve_polarization_task(sample_punchdata_list):
    """
    Test the resolve polarization prefect flow using a test harness
    """

    with disable_run_logger():
        output_punchdata_list = resolve_polarization_task.fn(sample_punchdata_list)

    assert all(isinstance(output_punchdata, PUNCHData) for output_punchdata in output_punchdata_list)
