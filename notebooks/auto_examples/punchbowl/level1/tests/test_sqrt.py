from datetime import UTC, datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect.logging import disable_run_logger
from pytest import fixture

from punchbowl.data import NormalizedMetadata
from punchbowl.data.units import split_ccd_array
from punchbowl.exceptions import DataValueWarning
from punchbowl.level1.sqrt import (
    decode_sqrt,
    decode_sqrt_by_table,
    decode_sqrt_data,
    decode_sqrt_simple,
    encode_sqrt,
    generate_decode_sqrt_table,
)


# Some test inputs
@fixture
def sample_punchdata():
    """
    Generate a sample PUNCH data object for testing
    """

    data = np.random.random([2048, 2048])
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.01, 0.01
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 0
    wcs.wcs.cname = "HPC lon", "HPC lat"
    wcs.array_shape = data.shape
    meta = NormalizedMetadata.load_template("PM1", "0")
    meta['DATE-OBS'] = str(datetime(2023, 1, 1, 0, 0, 1))

    punchdata_obj = NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)

    punchdata_obj.meta['RAWBITS'] = 16
    punchdata_obj.meta['COMPBITS'] = 10
    punchdata_obj.meta['GAINBTM'] = 1.0/4.9
    punchdata_obj.meta['GAINTOP'] = 1.0/4.9
    punchdata_obj.meta['OFFSET'] = 100

    return punchdata_obj


def test_encoding():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2 ** 16)

    encoded_arr = encode_sqrt(arr, from_bits=16, to_bits=10)

    assert encoded_arr.shape == arr.shape
    assert np.max(encoded_arr) <= 2 ** 10


def test_decoding():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2 ** 10)

    decoded_arr = decode_sqrt_simple(arr, from_bits=10, to_bits=16)

    assert decoded_arr.shape == arr.shape
    assert np.max(decoded_arr) <= 2 ** 16


def test_second_order_table():
    ccd_bits=16
    encoded_bits=10

    table1 = generate_decode_sqrt_table(from_bits = ccd_bits, to_bits = encoded_bits, first_order=True)
    table2 = generate_decode_sqrt_table(from_bits = ccd_bits, to_bits = encoded_bits, first_order=False)

    table_diff = table1 - table2

    assert table_diff.max() != 0
    assert table_diff.min() != 0

    data = np.arange(2**encoded_bits)

    data1 = decode_sqrt_by_table(data, table1)
    data2 = decode_sqrt_by_table(data, table2)

    data_diff = np.abs(data1 - data2)

    assert data_diff.max() < 2
    assert data_diff.max() != 0


def test_decoding_exceed_table():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2 ** 10) + 1

    with pytest.warns(DataValueWarning):
        decoded_arr = decode_sqrt(arr, from_bits=16, to_bits=10)
        assert decoded_arr.shape == arr.shape
        assert np.max(decoded_arr) <= 2 ** 16


@pytest.mark.parametrize('from_bits, to_bits', [(16, 10), (16, 11), (16, 12)])
def test_encode_then_decode(from_bits, to_bits):
    arr_dim = 2048
    ccd_gain_bottom = 1.0 / 4.9  # DN/electron
    ccd_gain_top = 1.0 / 4.9  # DN/electron
    ccd_offset = 100  # DN
    ccd_read_noise = 17  # DN

    original_arr = (np.random.random([arr_dim, arr_dim]) * (2 ** from_bits)).astype(int)

    encoded_arr = encode_sqrt(original_arr, from_bits, to_bits)
    decoded_arr = decode_sqrt(encoded_arr,
                              from_bits=from_bits,
                              to_bits=to_bits,
                              ccd_gain_bottom=ccd_gain_bottom,
                              ccd_gain_top=ccd_gain_top,
                              ccd_offset=ccd_offset,
                              ccd_read_noise=ccd_read_noise)

    ccd_gain = split_ccd_array(original_arr.shape, ccd_gain_bottom, ccd_gain_top)

    noise_tolerance = np.sqrt(original_arr / ccd_gain) * ccd_gain

    test_coords = np.where(original_arr > 150)

    assert np.all(np.abs(original_arr[test_coords] - decoded_arr[test_coords]) <= noise_tolerance[test_coords])


def test_decode_sqrt_data_task(sample_punchdata):
    """
    Test the decode_sqrt_data prefect task using a test harness
    """

    with disable_run_logger():
        output_punchdata = decode_sqrt_data.fn(sample_punchdata, overwrite_table=True)
        assert isinstance(output_punchdata, NDCube)
        assert output_punchdata.data.shape == (2048, 2048)
