from prefect.testing.utilities import prefect_test_harness

from punchbowl.data import PUNCHData
from punchbowl.level1.flow import level1_core_flow
from punchbowl.tests.test_data import sample_data_random, sample_punchdata

# def test_core_flow_runs_with_filenames(sample_punchdata):
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         input_name = tmpdirname + "/test_input.fits"
#         output_name = tmpdirname + "/test_output.fits"
#         sample_punchdata(shape=(2048, 2048)).write(input_name)
#         with prefect_test_harness():
#             level1_core_flow(input_name, output_filename=output_name)
#         output = PUNCHData.from_fits(output_name)
#         assert isinstance(output, PUNCHData)


def test_core_flow_runs_with_objects(sample_punchdata):
    """Simply tests that the core flow runs with objects"""
    with prefect_test_harness():
        output = level1_core_flow(sample_punchdata(shape=(2048, 2048)))
    assert isinstance(output[0], PUNCHData)
