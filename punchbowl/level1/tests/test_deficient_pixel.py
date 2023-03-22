# Core Python imports
import pathlib

# Third party imports
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from prefect.logging import disable_run_logger

# punchbowl imports
from punchbowl.data import NormalizedMetadata, PUNCHData
from punchbowl.level1.deficient_pixel import (
    remove_deficient_pixels,
)

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@pytest.fixture()
def sample_bad_pixel_map(shape: tuple = (2048, 2048), n_bad_pixels: int = 20) -> np.ndarray:
    """
    Generate some random data for testing
    """
    bad_pixel_map = np.ones(shape)

    x_coords = np.fix(np.random.random(n_bad_pixels) * shape[0]).astype(int)
    y_coords = np.fix(np.random.random(n_bad_pixels) * shape[1]).astype(int)

    bad_pixel_map[x_coords, y_coords] = 0

    bad_pixel_map = bad_pixel_map.astype(int)

    uncertainty = StdDevUncertainty(bad_pixel_map)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return PUNCHData(data=bad_pixel_map, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.fixture()
def perfect_pixel_map(shape: tuple = (2048, 2048)) -> np.ndarray:
    """
    Generate some random data for testing
    """
    bad_pixel_map = np.ones(shape)

    bad_pixel_map = bad_pixel_map.astype(int)

    uncertainty = StdDevUncertainty(bad_pixel_map)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return PUNCHData(data=bad_pixel_map, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.fixture()
def one_bad_pixel_map(shape: tuple = (2048, 2048)) -> np.ndarray:
    """
    Generate pixel map with one bad pixel at 100, 100
    """
    bad_pixel_map = np.ones(shape)

    bad_pixel_map = bad_pixel_map.astype(int)
    
    bad_pixel_map[100,100]=0
    
    uncertainty = StdDevUncertainty(bad_pixel_map)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return PUNCHData(data=bad_pixel_map, uncertainty=uncertainty, wcs=wcs, meta=meta)

@pytest.fixture()
def nine_bad_pixel_map(shape: tuple = (2048, 2048)) -> np.ndarray:
    """
    Generate pixel map with one bad pixel at 100, 100
    """
    bad_pixel_map = np.ones(shape)

    bad_pixel_map = bad_pixel_map.astype(int)
    
    bad_pixel_map[100,100]=0
    bad_pixel_map[100,101]=0
    bad_pixel_map[100,102]=0
    bad_pixel_map[101,100]=0
    bad_pixel_map[101,101]=0
    bad_pixel_map[101,102]=0
    bad_pixel_map[102,100]=0
    bad_pixel_map[102,101]=0
    bad_pixel_map[102,102]=0


    uncertainty = StdDevUncertainty(bad_pixel_map)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return PUNCHData(data=bad_pixel_map, uncertainty=uncertainty, wcs=wcs, meta=meta)




@pytest.fixture()
def increasing_pixel_data(shape: tuple = (2048, 2048)) -> np.ndarray:
    """
    Generate data of increasing values for testing; data[0,100]=0.0, data[100,0]=100.0
    """
    data = np.ones(shape)
    for iStep in range(2048):
        data[iStep,:]=iStep
    
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return PUNCHData(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.fixture()
def sample_punchdata(shape: tuple = (2048, 2048)) -> PUNCHData:
    """
    Generate a sample PUNCHData object for testing
    """

    data = np.random.random(shape)
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return PUNCHData(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.mark.prefect_test()
def test_remove_deficient_pixels(sample_punchdata: PUNCHData, sample_bad_pixel_map: PUNCHData) -> None:
    """
    Test the remove_deficient_pixels prefect flow using a test harness, providing a filename
    """
    with disable_run_logger():
        flagged_punchdata = remove_deficient_pixels.fn(sample_punchdata,
                                                       sample_bad_pixel_map)

        assert isinstance(flagged_punchdata, PUNCHData) 
        #assert np.all(flagged_punchdata.uncertainty[np.where(sample_pixel_map == 1)].array == np.inf)


@pytest.mark.prefect_test()
def test_nan_input(sample_punchdata: PUNCHData, sample_bad_pixel_map: PUNCHData) -> None:
    """
    The module output is tested when NaN data points are included in the input PUNCHData object. Test for no errors.
    """

    input_data = sample_punchdata
    input_data.data[42, 42] = np.nan

    with disable_run_logger():
        flagged_punchdata = remove_deficient_pixels.fn(input_data, sample_bad_pixel_map)

        assert isinstance(flagged_punchdata, PUNCHData)
        assert np.all(flagged_punchdata.uncertainty[np.where(sample_bad_pixel_map == 1)].array == np.inf)



@pytest.mark.prefect_test()
def test_data_loading(sample_punchdata: PUNCHData, perfect_pixel_map: PUNCHData) -> None:
    """
    A specific observation is provided. The module loads it as a PUNCHData object. 
    No bad data points, in same as out. uncertainty should be the same in and out.
    """
    with disable_run_logger():
        deficient_punchdata = remove_deficient_pixels.fn(sample_punchdata, perfect_pixel_map)

    assert isinstance(deficient_punchdata, PUNCHData)
    assert np.all(deficient_punchdata.data == sample_punchdata.data)
    assert np.all(deficient_punchdata.uncertainty.array == sample_punchdata.uncertainty.array)


@pytest.mark.prefect_test()
def test_artificial_pixel_map(sample_punchdata: PUNCHData, sample_bad_pixel_map: PUNCHData) -> None: 
    """
    A known artificial bad pixel map is ingested. The output flags are tested against the input map.
    """

    with disable_run_logger():
        flagged_punchdata = remove_deficient_pixels.fn(sample_punchdata, sample_bad_pixel_map)

        assert isinstance(flagged_punchdata, PUNCHData)
        assert np.all(flagged_punchdata.uncertainty[np.where(sample_bad_pixel_map == 1)].array == np.inf)


@pytest.mark.prefect_test()
def test_data_window_1(increasing_pixel_data: PUNCHData, one_bad_pixel_map: PUNCHData) -> None:
    """
    dataset of increasing values passed in, a bad pixel map is passed in 
    """
    with disable_run_logger():
        deficient_punchdata = remove_deficient_pixels.fn(increasing_pixel_data, one_bad_pixel_map)

    assert isinstance(deficient_punchdata, PUNCHData)
    assert (deficient_punchdata.data[5,5] == increasing_pixel_data.data[5,5])
    assert (deficient_punchdata.uncertainty.array[5,5] == increasing_pixel_data.uncertainty.array[5,5])
    assert (deficient_punchdata.data[100,100] == 100)

@pytest.mark.prefect_test()
def test_mean_data_window_1(increasing_pixel_data: PUNCHData, one_bad_pixel_map: PUNCHData) -> None:
    """
    dataset of increasing values passed in, a bad pixel map is passed in 
    """
    with disable_run_logger():
        deficient_punchdata = remove_deficient_pixels.fn(increasing_pixel_data, one_bad_pixel_map, method='mean')

    assert isinstance(deficient_punchdata, PUNCHData)
    assert (deficient_punchdata.data[5,5] == increasing_pixel_data.data[5,5])
    assert (deficient_punchdata.uncertainty.array[5,5] == increasing_pixel_data.uncertainty.array[5,5])
    assert (deficient_punchdata.data[100,100] == 100)

@pytest.mark.prefect_test()
def test_data_window_9(increasing_pixel_data: PUNCHData, nine_bad_pixel_map: PUNCHData) -> None:
    """
    dataset of increasing values passed in, a bad pixel map is passed in 
    """
    with disable_run_logger():
        deficient_punchdata = remove_deficient_pixels.fn(increasing_pixel_data, nine_bad_pixel_map)

    assert isinstance(deficient_punchdata, PUNCHData)
    assert (deficient_punchdata.data[5,5] == increasing_pixel_data.data[5,5])
    assert (deficient_punchdata.uncertainty.array[5,5] == increasing_pixel_data.uncertainty.array[5,5])
    assert (deficient_punchdata.data[101,101] == 101)  

@pytest.mark.prefect_test()
def test_mean_data_window_9(increasing_pixel_data: PUNCHData, nine_bad_pixel_map: PUNCHData) -> None:
    """
    dataset of increasing values passed in, a bad pixel map is passed in 
    """
    with disable_run_logger():
        deficient_punchdata = remove_deficient_pixels.fn(increasing_pixel_data, nine_bad_pixel_map, method='mean')

    assert isinstance(deficient_punchdata, PUNCHData)
    assert (deficient_punchdata.data[5,5] == increasing_pixel_data.data[5,5])
    assert (deficient_punchdata.uncertainty.array[5,5] == increasing_pixel_data.uncertainty.array[5,5])
    assert (deficient_punchdata.data[101,101] == 101)