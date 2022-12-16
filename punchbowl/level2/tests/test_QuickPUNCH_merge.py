# Core Python imports
import pytest
from pytest import fixture
import pathlib

# Third party imports
import numpy as np

from astropy.wcs import WCS

from ndcube import NDCube

from prefect.testing.utilities import prefect_test_harness

# punchbowl imports
from punchbowl.level2.QuickPUNCH_merge import reproject_array, mosaic, \
    quickpunch_merge_flow

from punchbowl.data import PUNCHData


# Some test inputs
@fixture
def sample_wcs(naxis=2, crpix=(0,0), crval=(0,0), cdelt=(1, 1), \
               ctype=("HPLN-ARC", "HPLT-ARC")) -> WCS:
    """
    Generate a WCS for testing
    """
    generated_wcs = WCS(naxis=naxis)

    generated_wcs.wcs.crpix = crpix
    generated_wcs.wcs.crval = crval
    generated_wcs.wcs.cdelt = cdelt
    generated_wcs.wcs.ctype = ctype

    return generated_wcs

@fixture
def sample_data(shape: tuple = (20,20)) -> np.ndarray:
    """
    Generate some random data for testing
    """

    return np.random.random(shape)

@fixture
def sample_ndcube(sample_data):
    data = sample_data
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLT-TAN", "HPLN-TAN"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.1, 0.1
    wcs.wcs.crpix = 0, 0
    wcs.wcs.crval = 1, 1
    wcs.wcs.cname = "HPC lat", "HPC lon"
    nd_obj = NDCube(data=data, wcs=wcs)
    return nd_obj


@fixture
def sample_punchdata(sample_ndcube):
    return PUNCHData(sample_ndcube)


@fixture
def sample_punchdata_list(sample_punchdata):
    return [sample_punchdata, sample_punchdata]


# core unit tests

# TODO - parameterize the test inputs below (input parameters to fixtures?)
# @pytest.mark.parametrize("input_array, input_wcs, output_wcs, output_shape",
#                         [(np.zeros([20,20]), sample_wcs, sample_wcs, (20,20))])
# def test_reproject_array(input_array, input_wcs, output_wcs, output_shape):


def test_reproject_array(sample_data, sample_wcs, output_shape=(20,20)):
    expected = sample_data
    actual = reproject_array(sample_data, sample_wcs, sample_wcs, output_shape)

    # assert np.allclose(actual, expected)
    assert actual.shape == expected.shape


def test_mosaic(sample_data, sample_wcs):

    expected = sample_data

    data_input = [sample_data, sample_data]
    uncert_input = [sample_data, sample_data]
    wcs_input = [sample_wcs, sample_wcs]
    wcs_output = sample_wcs
    shape_output = (20,20)

    (actual_data, actual_uncert) = mosaic(data_input, uncert_input, wcs_input, \
                                          wcs_output, shape_output)

    assert actual_data.shape == expected.shape


#@pytest.mark.prefect_test
def test_quickpunch_merge_flow(sample_punchdata_list):
    with prefect_test_harness():
        output_punchdata = quickpunch_merge_flow(sample_punchdata_list)
        assert isinstance(output_punchdata, PUNCHData)
