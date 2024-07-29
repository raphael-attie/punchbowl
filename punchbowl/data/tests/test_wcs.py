import os
from datetime import datetime

import astropy
import astropy.units as u
import numpy as np
from astropy.coordinates import GCRS, EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from ndcube import NDCube
from sunpy.coordinates import frames

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.wcs import calculate_celestial_wcs_from_helio, calculate_helio_wcs_from_celestial, load_trefoil_wcs

_ROOT = os.path.abspath(os.path.dirname(__file__))


def test_sun_location():
    time_current = Time(datetime.utcnow())

    skycoord_sun = astropy.coordinates.get_sun(Time(datetime.utcnow()))

    skycoord_origin = SkyCoord(0*u.deg, 0*u.deg,
                              frame=frames.Helioprojective,
                              obstime=time_current,
                              observer='earth')

    with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
        skycoord_origin_celestial = skycoord_origin.transform_to(GCRS)

    with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
        assert skycoord_origin_celestial.separation(skycoord_sun) < 1 * u.arcsec
        assert skycoord_origin.separation(skycoord_sun) < 1 * u.arcsec


def test_wcs_many_point_2d_check():
    m = NormalizedMetadata.load_template("CTM", "2")
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)

    sun_radec = get_sun(date_obs)
    m['CRVAL1A'] = sun_radec.ra.to(u.deg).value
    m['CRVAL2A'] = sun_radec.dec.to(u.deg).value
    h = m.to_fits_header()
    d = NDCube(np.ones((4096, 4096), dtype=np.float32), WCS(h, key='A'), m)

    # we're at the center of the Earth so let's try that
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    wcs_celestial = d.wcs

    wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, d.data.shape)

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int)], axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []

    for c_pix, c_celestial, c_helio in zip(input_coords, points_celestial, points_helio):
        skycoord_celestial = SkyCoord(c_celestial[0] * u.deg, c_celestial[1] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer=test_gcrs,
                                      obsgeoloc=test_gcrs.cartesian,
                                      obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      distance=test_gcrs.hcrs.distance
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective)
        output_coords.append(wcs_helio.all_world2pix(intermediate.data.lon.to(u.deg).value,
                                                     intermediate.data.lat.to(u.deg).value, 0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)
    assert np.mean(distances) < 0.1


def test_wcs_many_point_3d_check():
    m = NormalizedMetadata.load_template("PSM", "3")
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)
    sun_radec = get_sun(date_obs)
    m['CRVAL1A'] = sun_radec.ra.to(u.deg).value
    m['CRVAL2A'] = sun_radec.dec.to(u.deg).value
    h = m.to_fits_header()
    d = NDCube(np.ones((2, 4096, 4096), dtype=np.float32), WCS(h, key='A'), m)

    # we're at the center of the Earth so let's try that
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    wcs_celestial = d.wcs
    wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, d.data.shape)

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int),
                             np.ones(npoints, dtype=int),], axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []
    for c_pix, c_celestial, c_helio in zip(input_coords, points_celestial, points_helio):
        skycoord_celestial = SkyCoord(c_celestial[0] * u.deg, c_celestial[1] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer=test_gcrs,
                                      obsgeoloc=test_gcrs.cartesian,
                                      obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      distance=test_gcrs.hcrs.distance
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective)
        output_coords.append(wcs_helio.all_world2pix(intermediate.data.lon.to(u.deg).value,
                                                     intermediate.data.lat.to(u.deg).value, 2, 0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)
    assert np.nanmean(distances) < 0.1


def test_load_trefoil_wcs():
    trefoil_wcs, trefoil_shape = load_trefoil_wcs()
    assert trefoil_shape == (4096, 4096)
    assert isinstance(trefoil_wcs, WCS)


def test_helio_celestial_wcs():
    header = fits.Header.fromtextfile(os.path.join(_ROOT, "example_header.txt"))

    wcs_helio = WCS(header)
    wcs_celestial = WCS(header, key='A')

    date_obs = Time(header['DATE-OBS'], format='isot', scale='utc')
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int),
                             np.ones(npoints, dtype=int),], axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []
    for c_pix, c_celestial, c_helio in zip(input_coords, points_celestial, points_helio):
        skycoord_celestial = SkyCoord(c_celestial[0] * u.deg, c_celestial[1] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer=test_gcrs,
                                      obsgeoloc=test_gcrs.cartesian,
                                      obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      distance=test_gcrs.hcrs.distance
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective)
        output_coords.append(wcs_helio.all_world2pix(intermediate.data.lon.to(u.deg).value,
                                                     intermediate.data.lat.to(u.deg).value, 2, 0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)

    assert np.nanmean(distances) < 0.1


def test_back_and_forth_wcs_from_celestial():
    m = NormalizedMetadata.load_template("CTM", "2")
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)

    sun_radec = get_sun(date_obs)
    m['CRVAL1A'] = sun_radec.ra.to(u.deg).value
    m['CRVAL2A'] = sun_radec.dec.to(u.deg).value
    h = m.to_fits_header()

    wcs_celestial = WCS(h, key='A')
    wcs_helio, p_angle = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, (10, 10))
    wcs_celestial_recovered = calculate_celestial_wcs_from_helio(wcs_helio, date_obs, (10, 10))

    assert np.allclose(wcs_celestial.wcs.crval, wcs_celestial_recovered.wcs.crval)
    assert np.allclose(wcs_celestial.wcs.crpix, wcs_celestial_recovered.wcs.crpix)
    assert np.allclose(wcs_celestial.wcs.pc, wcs_celestial_recovered.wcs.pc)
    assert np.allclose(wcs_celestial.wcs.cdelt, wcs_celestial_recovered.wcs.cdelt)


def test_back_and_forth_wcs_from_helio():
    m = NormalizedMetadata.load_template("CTM", "2")
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)
    m['CRVAL1'] = 0.0
    m['CRVAL2'] = 0.0
    h = m.to_fits_header()

    wec_helio = WCS(h, key=' ')
    wcs_celestial = calculate_celestial_wcs_from_helio(wec_helio, date_obs, (10, 10))
    wcs_helio_recovered, p_angle = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, (10, 10))

    assert np.allclose(wec_helio.wcs.crval, wcs_helio_recovered.wcs.crval)
    assert np.allclose(wec_helio.wcs.crpix, wcs_helio_recovered.wcs.crpix)
    assert np.allclose(wec_helio.wcs.pc, wcs_helio_recovered.wcs.pc)
    assert np.allclose(wec_helio.wcs.cdelt, wcs_helio_recovered.wcs.cdelt)
