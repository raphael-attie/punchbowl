import os
import pathlib

import pytest
from ndcube import NDCube

from punchbowl.data.punch_io import write_ndcube_to_fits
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level1.flow import level1_early_core_flow, level1_late_core_flow

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()

def test_early_core_flow_runs_with_filenames(sample_ndcube, tmpdir):
    input_name = os.path.join(tmpdir, "test_input.fits")
    output_name = os.path.join(tmpdir, "test_output.fits")
    sample_data = sample_ndcube(shape=(10, 10), code="PM1", level="1")
    sample_data.meta["RAWBITS"] = 16
    sample_data.meta["COMPBITS"] = 10
    sample_data.meta["GAINBTM"] = 4.9
    sample_data.meta["GAINTOP"] = 4.9
    sample_data.meta["OFFSET"] = 100
    sample_data.meta["EXPTIME"] = 49
    sample_data.meta["FILENAME"] = 'test_input.fits'
    write_ndcube_to_fits(sample_data, input_name)

    quartic_coefficient_path = THIS_DIRECTORY / "data" / "test_quartic_coeffs.fits"
    vignetting_path = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"
    output = level1_early_core_flow([input_name],
                              quartic_coefficient_path=quartic_coefficient_path,
                              vignetting_function_path=vignetting_path)
    assert isinstance(output[0], NDCube)
    assert os.path.exists(output_name[0])



def test_early_core_flow_runs_with_objects_and_calibration_files(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CR1", level="1")
    cube.meta["RAWBITS"] = 16
    cube.meta["COMPBITS"] = 10
    cube.meta["GAINBTM"] = 4.9
    cube.meta["GAINTOP"] = 4.9
    cube.meta["OFFSET"] = 100
    cube.meta["EXPTIME"] = 49
    cube.meta["FILENAME"] = 'test_input.fits'
    quartic_coefficient_path = THIS_DIRECTORY / "data" / "test_quartic_coeffs.fits"
    vignetting_path = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    output = level1_early_core_flow([cube],
                              quartic_coefficient_path=quartic_coefficient_path,
                              vignetting_function_path=vignetting_path)
    assert isinstance(output[0], NDCube)


def test_late_core_flow_runs_with_filenames(sample_ndcube, tmpdir):
    input_name = os.path.join(tmpdir, "test_input.fits")
    output_name = os.path.join(tmpdir, "test_output.fits")
    sample_data = sample_ndcube(shape=(10, 10), code="XM1", level="1")
    write_ndcube_to_fits(sample_data, input_name)

    stray_light_before = THIS_DIRECTORY / "data" / "PUNCH_L1_SM1_20240222163425_v1.fits"
    stray_light_after = THIS_DIRECTORY / "data" / "PUNCH_L1_SM1_20240222163425_v1.fits"

    output = level1_late_core_flow([input_name],
                                   output_filename=[output_name],
                                   stray_light_before_path=stray_light_before,
                                   stray_light_after_path=stray_light_after,
                                   do_align=False)
    assert isinstance(output[0], NDCube)
    assert os.path.exists(output_name[0])
    assert output[0].meta['TYPECODE'].value == 'PM'


def test_quick_core_flow_runs_with_filenames(sample_ndcube, tmpdir):
    input_name = os.path.join(tmpdir, "test_input.fits")
    output_name = os.path.join(tmpdir, "test_output.fits")
    sample_data = sample_ndcube(shape=(10, 10), code="XM1", level="1")
    write_ndcube_to_fits(sample_data, input_name)

    stray_light_before = THIS_DIRECTORY / "data" / "PUNCH_L1_SM1_20240222163425_v1.fits"
    stray_light_after = THIS_DIRECTORY / "data" / "PUNCH_L1_SM1_20240222163425_v1.fits"

    output = level1_late_core_flow([input_name],
                                   output_filename=[output_name],
                                   stray_light_before_path=stray_light_before,
                                   stray_light_after_path=stray_light_after,
                                   do_align=False,
                                   output_as_Q_file=True,
                                   )
    assert isinstance(output[0], NDCube)
    assert os.path.exists(output_name[0])
    assert output[0].meta['TYPECODE'].value == 'QM'
