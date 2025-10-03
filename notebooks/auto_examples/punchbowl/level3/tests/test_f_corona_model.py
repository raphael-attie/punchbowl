# Core Python imports
import os
import pathlib
from glob import glob

# Third party imports
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

# punchbowl imports
from punchbowl.data import NormalizedMetadata
from punchbowl.data.meta import MetaField
from punchbowl.exceptions import InvalidDataError
from punchbowl.level3.f_corona_model import subtract_f_corona_background, subtract_f_corona_background_task

TEST_DIRECTORY = pathlib.Path(__file__).parent.resolve()
TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH = os.path.join(TESTDATA_DIR, "L0_CL1_20211111070246_PUNCHData.fits")

@pytest.fixture()
def sample_data_list():
    number_elements = 25
    data_list = []
    for iStep in range(number_elements):
        data_list.append(SAMPLE_FITS_PATH)
    return data_list


@pytest.fixture()
def one_data(shape: tuple = (2048, 2048)) -> np.ndarray:
    """
    Generate some random data for testing
    """
    data = np.ones(shape)

    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75
    wcs.array_shape = shape

    meta = NormalizedMetadata({"SECTION": {
        "TYPECODE": MetaField("TYPECODE", "", "CL", str, True, True, ""),
        "LEVEL": MetaField("LEVEL", "", "1", str, True, True, ""),
        "OBSRVTRY": MetaField("OBSRVTRY", "", "0", str, True, True, ""),
        "DATE-OBS": MetaField("DATE-OBS", "", "2008-01-03T04:57:00", str, True, True, "")}})
    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)

@pytest.fixture()
def observation_data(shape: tuple = (2048, 2048)) -> np.ndarray:
    """
    Generate some random data for testing
    """
    data = np.ones(shape)

    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75
    wcs.array_shape = shape

    meta = NormalizedMetadata({"SECTION": {
        "TYPECODE": MetaField("TYPECODE", "", "CL", str, True, True, ""),
        "LEVEL": MetaField("LEVEL", "", "1", str, True, True, ""),
        "OBSRVTRY": MetaField("OBSRVTRY", "", "0", str, True, True, ""),
        "DATE-OBS": MetaField("DATE-OBS", "", "2008-01-03T08:57:00", str, True, True, "")}})
    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.fixture()
def zero_data(shape: tuple = (2048, 2048)) -> np.ndarray:
    """
    Generate some random data for testing
    """
    data = np.zeros(shape)

    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75
    wcs.array_shape = shape

    meta = NormalizedMetadata({"SECTION": {
        "TYPECODE": MetaField("TYPECODE", "", "CL", str, True, True, ""),
        "LEVEL": MetaField("LEVEL", "", "1", str, True, True, ""),
        "OBSRVTRY": MetaField("OBSRVTRY", "", "0", str, True, True, ""),
        "DATE-OBS": MetaField("DATE-OBS", "", "2008-01-03T12:57:00", str, True, True, "")}})
    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.fixture()
def incorrect_shape_data(shape: tuple = (512, 512)) -> np.ndarray:
    """
    Generate some random data for testing
    """
    data = np.zeros(shape)

    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 256, 256
    wcs.wcs.crval = 0, 24.75
    wcs.array_shape = shape

    meta = NormalizedMetadata({"SECTION": {
        "TYPECODE": MetaField("TYPECODE", "", "CL", str, True, True, ""),
        "LEVEL": MetaField("LEVEL", "", "1", str, True, True, ""),
        "OBSRVTRY": MetaField("OBSRVTRY", "", "0", str, True, True, ""),
        "DATE-OBS": MetaField("DATE-OBS", "", "2008-01-03T08:57:00", str, True, True, "")}})

    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


def test_basic_subtraction(observation_data: NDCube, one_data: NDCube, zero_data: NDCube) -> None:
    """
    dataset of increasing values passed in, a bad pixel map is passed in
    """
    subtraction_punchdata = subtract_f_corona_background(observation_data, one_data, zero_data)
    assert isinstance(subtraction_punchdata, NDCube)
    assert np.all(subtraction_punchdata.data == 0.5)

def test_flipped_dates_subtraction_fails(observation_data: NDCube, one_data: NDCube, zero_data: NDCube) -> None:
    """
    dataset of increasing values passed in, a bad pixel map is passed in
    """
    with pytest.raises(InvalidDataError):
        _ = subtract_f_corona_background(observation_data, zero_data, one_data)


def test_after_is_before_subtraction_fails(observation_data: NDCube, one_data: NDCube, zero_data: NDCube) -> None:
    """
    dataset of increasing values passed in, a bad pixel map is passed in
    """
    with pytest.raises(InvalidDataError):
        _ = subtract_f_corona_background(observation_data, one_data, one_data)


def test_different_array_size_subtraction(incorrect_shape_data: NDCube, zero_data: NDCube) -> None:
    """
    dataset of increasing values passed in, a bad pixel map is passed in
    """
    with pytest.raises(Exception):
        subtract_f_corona_background_task(incorrect_shape_data, zero_data)
