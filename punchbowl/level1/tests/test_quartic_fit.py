import pathlib
import numpy as np
import pytest

from ndcube import NDCube
from pytest import fixture
from datetime import datetime
from prefect.logging import disable_run_logger
from astropy.io import fits

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level1.quartic_fit import (
    create_coefficient_image,
    create_constant_quartic_coefficients,
    photometric_calibration,
    perform_quartic_fit_task
)


from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
    DataShapeError
)

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()

@fixture
def fake_one_image(shape=(10, 10)):
    """
    Parameters
    ----------
    shape : tuple
        the shape of the image to create

    Returns
    -------
    A fake image of the given shape with values of all 1

    """
    return np.ones(shape)


@fixture
def quartic_coefficients_image():
    """
    Returns
    -------
        Quartic coefficient image with flat_coefficients of [0, 1, 2, 3, 4]
    """
    flat_coeffs = np.array([0, 1, 2, 3, 4])
    image_shape = (10, 10)
    return create_coefficient_image(flat_coeffs, image_shape)


def test_create_constant_coefficient_image():
    image_shape = (10, 20)
    coeffs = create_constant_quartic_coefficients(image_shape)
    assert coeffs.shape[:-1] == image_shape
    assert coeffs.shape[-1] == 5
    for i in [0, 1, 2, 4]:
        assert np.all(coeffs[:, :, i] == 0)
    assert np.all(coeffs[:, :, 3] == 1)


def test_create_coeff_image_with_only_one_coeff():
    coeffs_image = create_coefficient_image(np.array([5]), (10, 10))
    assert coeffs_image.shape == (
        10,
        10,
        1,
    ), "Failed to make a coefficients image with only one coefficient"


def test_create_coeff_image_with_two_coeff():
    coeffs_image = create_coefficient_image(np.array([5, 6]), (10, 10))
    assert coeffs_image.shape == (
        10,
        10,
        2,
    ), "Failed to make a coefficients image with two coefficients"


def test_create_coefficient_image(quartic_coefficients_image):
    assert quartic_coefficients_image.shape == (
        10,
        10,
        5,
    ), "Quartic coefficient image shape is wrong."
    for i in range(quartic_coefficients_image.shape[-1]):
        assert np.all(
            quartic_coefficients_image[:, :, i] == i
        ), "Quartic coefficient image coefficients were not constructed properly."


@pytest.mark.parametrize("constant", [1, 2, 3])
def test_calibration_with_single_constant_coeff(
    fake_one_image, constant, shape=(10, 10)
):
    coeffs_image = create_coefficient_image(np.array([constant]), shape)
    calibrated = photometric_calibration(fake_one_image, coeffs_image)
    assert np.all(
        calibrated == constant
    ), "Photometric calibration with a single constant failed"


def test_calibration_with_all_one_coefficients(fake_one_image, shape=(10, 10)):
    coeffs_image = create_coefficient_image(np.array([1, 1, 1, 1, 1]), shape)
    calibrated = photometric_calibration(fake_one_image, coeffs_image)
    assert np.all(
        calibrated == 5
    ), "Photometric calibration with all one coefficients failed"


def test_calibration_with_range_coefficients(fake_one_image, shape=(10, 10)):
    coeffs_image = create_coefficient_image(np.array([1, 2, 3, 4, 5]), shape)
    calibrated = photometric_calibration(fake_one_image, coeffs_image)
    assert np.all(
        calibrated == 15
    ), "Photometric calibration with range coefficients failed"


@pytest.mark.prefect_test()
def test_check_calibration_time_delta_warning(sample_ndcube) -> None:
    """
    If the time between the data of interest and the calibration file is too 
    great, then a warning is raised.
    """
    sample_data = sample_ndcube(shape=(10, 10))
    sample_data.meta['DATE-OBS'].value = str(datetime(2020, 2, 22, 16, 0, 1))
    quartic_fit_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_FQ1_20240101000000.fits"
    with disable_run_logger():
        with pytest.warns(LargeTimeDeltaWarning):
            corrected_punchdata = perform_quartic_fit_task.fn(sample_data, quartic_fit_filename)
            assert isinstance(corrected_punchdata, NDCube)

@pytest.mark.prefect_test()
def test_no_quartic_fit_file(sample_ndcube) -> None:
    """
    An invalid quartic fit file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    quartic_fit_filename = None
    with disable_run_logger():
        corrected_punchdata = perform_quartic_fit_task.fn(sample_data, quartic_fit_filename)
        assert isinstance(corrected_punchdata, NDCube)
        assert corrected_punchdata.meta.history[0].comment == 'Quartic fit skipped'

@pytest.mark.prefect_test()
def test_invalid_vignetting_file(sample_ndcube) -> None:
    """
    An invalid quartic fit file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    quartic_fit_filename = THIS_DIRECTORY / "data" / "bogus_filename.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = perform_quartic_fit_task.fn(sample_data, quartic_fit_filename)

@pytest.mark.prefect_test()
def test_invalid_polarization_state(sample_ndcube) -> None:
    """
    Check that a mismatch between polarization states in the quartic fit function and data raises an error.
    """
    sample_data = sample_ndcube(shape=(10, 10), code="FQ3", level="1")
    quartic_fit_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_FQ1_20240101000000.fits"
    sample_data.meta['OBSLAYR1'].value = str('Polar_Z')
    with disable_run_logger():
        with pytest.warns(IncorrectPolarizationStateWarning):
            corrected_punchdata = perform_quartic_fit_task.fn(sample_data, quartic_fit_filename)
            assert isinstance(corrected_punchdata, NDCube)

@pytest.mark.prefect_test()
def test_invalid_telescope(sample_ndcube) -> None:
    """
    Check that a mismatch between telescopes in the vignetting function and data raises an error.
    """

    sample_data = sample_ndcube(shape=(10, 10))
    sample_data.meta['TELESCOP'].value = 'PUNCH-2'
    quartic_fit_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_FQ1_20240101000000.fits"

    with disable_run_logger():
        with pytest.warns(IncorrectTelescopeWarning):
            corrected_punchdata = perform_quartic_fit_task.fn(sample_data, quartic_fit_filename)
            assert isinstance(corrected_punchdata, NDCube)

@pytest.mark.prefect_test()
def test_invalid_data_file(sample_ndcube) -> None:
    """
    An invalid quartic fit file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    quartic_fit_filename = THIS_DIRECTORY / "non_existent_file.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = perform_quartic_fit_task.fn(sample_data, quartic_fit_filename)

@pytest.mark.prefect_test()
def test_quartic_fit_correction(sample_ndcube) -> None:
    """
    A valid quartic fit file should be provided. Check that a corrected PUNCHData object is generated.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    quartic_fit_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_FQ1_20240101000000.fits"

    with disable_run_logger():
        corrected_punchdata = perform_quartic_fit_task.fn(sample_data, quartic_fit_filename)

    assert isinstance(corrected_punchdata, NDCube)


@pytest.mark.prefect_test()
def test_data_fit_shape(sample_ndcube) -> None:
    """
    A valid data file should be provided. Check that a corrected PUNCHData object is generated.
    """

    sample_data = sample_ndcube(shape=(10, 10, 10), code="CR1", level="0")
    quartic_fit_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_FQ1_20240101000000.fits"

    with disable_run_logger():
        with pytest.raises(DataShapeError):
            corrected_punchdata = perform_quartic_fit_task.fn(sample_data, quartic_fit_filename)

@pytest.mark.prefect_test()
def test_quartic_fit_shape(sample_ndcube) -> None:
    """
    A valid quartic fit file should be provided. Check that a corrected PUNCHData object is generated.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    quartic_fit_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_FQ1_20240102000000.fits"

    with disable_run_logger():
        with pytest.raises(DataShapeError):
            corrected_punchdata = perform_quartic_fit_task.fn(sample_data, quartic_fit_filename)
