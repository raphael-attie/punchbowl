import numpy as np
from astropy.io import fits
import glob
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os


def read_fits(filepath: str, hdu: int = 1) -> np.ndarray:
    """
    Read FITS image array.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    hdu : int, optional
        Index of the header data unit storing the data array, by default 1.

    Returns
    -------
    np.ndarray
        2D image array from the specified FITS file.
    """
    with fits.open(filepath) as hdul:
        hdul.verify('fix')
        data = hdul[hdu].data
    return data


def read_header(filepath: str, hdu: int = 1) -> fits.header.Header:
    """
    Read the header from a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    hdu : int, optional
        Index of the header data unit containing the header information, by default 1.

    Returns
    -------
    fits.header.Header
        Header object from the specified FITS file.
    """
    with fits.open(filepath) as hdul:
        hdul.verify('fix')
        header = hdul[hdu].header
    return header


def calc_ylims(ycen_band_rs: np.ndarray, band_width: float, header: fits.header.Header) -> [int, int]:
    """
    Convert y-coordinates of lower and upper row of bands to array indices for slicing.

    Parameters
    ----------
    ycen_band_rs : np.ndarray
        y-coordinates of center of band in solar radii.

    band_width : float
        Half-width of the band in solar radii.

    header : fits.header.Header
        FITS header containing image information.

    Returns
    -------
    list
        Lower and upper Numpy array indices of the band.
    """
    y0_image_px = 1
    y0_cen_px = header['CRPIX2']
    y0_cen_rs = header['CRVAL2']
    px_scale = header['CDELT2']
    origin_rs = y0_cen_rs + (y0_image_px - y0_cen_px) * px_scale
    ylo_band_idx = ((ycen_band_rs - band_width) - origin_rs) / px_scale
    yhi_band_idx = ((ycen_band_rs + band_width) - origin_rs) / px_scale
    return [ylo_band_idx, yhi_band_idx]


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize and preprocess FITS image by removing bad values and scaling.

    Parameters
    ----------
    image : np.ndarray
        Input FITS image array.

    Returns
    -------
    np.ndarray
        Preprocessed image array.
    """
    image[np.abs(image) > 100] = 0
    image[~np.isfinite(image)] = 0
    image -= np.mean(image, axis=0)
    return image / np.sqrt(np.mean(image ** 2, axis=0))


def calculate_cross_correlation(a: np.ndarray, b: np.ndarray, offsets: np.ndarray, dr: int, of0: int) -> np.ndarray:
    """
    Perform cross-correlation for a range of offsets.

    Parameters
    ----------
    a : np.ndarray
        First image array for correlation.

    b : np.ndarray
        Second, time-offseted image array for correlation.

    offsets : np.ndarray
        Array of pixel offsets to iterate over for cross-correlation.

    dr : int
        Pixel offset increment between samples.

    of0 : int
        Central offset to start correlation from.

    Returns
    -------
    np.ndarray
        Accumulated cross-correlation array over all offsets.
    """
    acc = np.zeros((len(offsets), a.shape[0], a.shape[1]), dtype=float)
    for jj, offset_index in enumerate(offsets):
        this_of = int(dr * (offset_index - (len(offsets) - 1) / 2)) + of0
        ofa = int(this_of / 2)
        ofb = int(this_of) - ofa

        if ofa < 0:
            aa = np.pad(a, ((0, -ofa), (0, 0)), mode='edge')[abs(ofa):a.shape[0] + abs(ofa), :]
        else:
            aa = np.pad(a, ((ofa, 0), (0, 0)), mode='edge')[:a.shape[0], :]

        if ofb < 0:
            bb = np.pad(b, ((-ofb, 0), (0, 0)), mode='edge')[:b.shape[0], :]
        else:
            bb = np.pad(b, ((0, ofb), (0, 0)), mode='edge')[ofb:b.shape[0] + ofb, :]

        acc[jj, :, :] += aa * bb

    return acc


def process_frame_pair(file_a: str, file_b: str, hdu: int, n_ofs: int, dr: int, of0: int) -> np.ndarray:
    """
    Process a pair of frames by reading, preprocessing, and calculating cross-correlation.

    Parameters
    ----------
    file_a : str
        File path to the first frame in the pair.

    file_b : str
        File path to the second, time-offset frame in the pair.

    hdu : int
        Index of the header data unit storing the data array.

    n_ofs : int
        Number of pixel offsets to use in cross-correlation.

    dr : int
        Pixel offset increment between samples.

    of0 : int
        Central offset to start correlation from.

    Returns
    -------
    np.ndarray
        Cross-correlation array for the given frame pair.
    """
    a = preprocess_image(read_fits(file_a, hdu=hdu))
    b = preprocess_image(read_fits(file_b, hdu=hdu))
    acc = calculate_cross_correlation(a, b, np.arange(n_ofs), dr, of0)

    return acc


def accumulate_cross_correlation_across_frames(files: list, hdu: int, dt: int, sparsity: int, n_ofs: int, dr: int,
                                               of0: int) -> np.ndarray:
    """
    Accumulate cross-correlation across frames in a list of FITS files.

    Parameters
    ----------
    files : list
        List of file paths to FITS files.

    hdu : int
        Index of the header data unit storing the data array.

    dt : int
        Frame offset (in frames) between time-offset image pairs.

    sparsity : int
        Interval between frames to skip when accumulating cross-correlation.

    n_ofs : int
        Number of pixel offsets to use in cross-correlation.

    dr : int
        Pixel offset increment between samples.

    of0 : int
        Central offset to start correlation from.

    Returns
    -------
    np.ndarray
        Accumulated cross-correlation array over all frames and offsets.
    """
    sample = read_fits(files[0], hdu=hdu)
    acc = np.zeros((n_ofs, sample.shape[0], sample.shape[1]), dtype=float)

    n = 0
    for i in range(0, len(files) - dt, dt * sparsity):
        print(f"Frame {i} vs frame {i + dt}")
        acc += process_frame_pair(files[i], files[i + dt], hdu, n_ofs, dr, of0)
        n += 1

    acc /= n

    return acc


def compute_all_bands(acc: np.ndarray, ycen_band_rs: np.ndarray, chw: float, header, nlatbins: int,
                      x_kps: np.ndarray):
    """
    Compute speed and sigma for all radial bands.

    Parameters
    ----------
    acc : np.ndarray
        Cross-correlation array accumulated across frames.

    ycen_band_rs : np.ndarray
        y-coordinates of band centers in solar radii.

    chw : float
        Half-width of each radial band in solar radii.

    header : fits.header.Header
        FITS header containing image information.

    nlatbins : int
        Number of angular bins for averaging.

    x_kps : np.ndarray
        Array mapping pixel offsets to speed in km/s.

    Returns
    -------
    tuple
        Tuple containing:
        - np.ndarray : Average speed per angular bin for each radial band.
        - np.ndarray : Sigma (standard deviation) of speed per angular bin for each radial band.
    """
    ylohi = calc_ylims(ycen_band_rs, chw, header)
    # Determine spike location (index of the correlation peak) in the cross-correlation array
    spike_location = np.where(x_kps < 0)[0].max() + 2

    avg_speeds = []
    sigmas = []
    for kk, (ylo, yhi) in enumerate(zip(*ylohi)):
        acc_k = acc[:, int(ylo):int(yhi) + 1, ...].mean(axis=1)
        latbinsize = acc_k.shape[1] // nlatbins
        avcor_rbins_theta = acc_k.reshape(acc_k.shape[0], latbinsize, nlatbins)

        speedmax_idx_per_thbin = np.array(
            [avcor_rbins_theta[spike_location:, :, i].argmax(axis=0) + spike_location for i in range(nlatbins)])
        speedmax_per_theta = x_kps[speedmax_idx_per_thbin]
        avg_speeds.append(speedmax_per_theta.mean(axis=1))
        sigmas.append(speedmax_per_theta.std(axis=1) / np.sqrt(latbinsize))

    return np.array(avg_speeds), np.array(sigmas)

def process_corr(files: list, hdu: int, dt: int, sparsity: int, dr: int,n_ofs: int, of0: int, nlatbins: int):
    """
    Process the cross-correlation across frames  in a list of FITS files with associated average speeds

    Parameters
    ----------

    files : list
        List of file paths to FITS files.

    hdu : int
        Index of the header data unit storing the data array.

    dt : float
        Time offset (in nb of frames) between for an image pair

    sparsity : int
        Interval between frames to skip when accumulating cross-correlation.

    dr : int
        Pixel offset increment between samples.

    n_ofs : int
        Number of pixel offsets to use in cross-correlation.

    of0: int
        Central offset to start correlation from.

    nlatbins : int
        Number of angular bins for averaging.

    Returns
    -------
    [np.ndarray, np.ndarray]
        Average speed and 1-sigma uncertainty over radius and angular bins

    """

    # Calculate speed mapping for offsets in km/s
    x_pix = dr * (np.arange(n_ofs) - (n_ofs - 1) / 2) + of0
    x_kps = x_pix * expected_kps_windspeed / of0
    # Accumulate cross-correlation across frames
    acc = accumulate_cross_correlation_across_frames(files, hdu, dt, sparsity, n_ofs, dr, of0)
    # Read header from the first file for y-limits calculation
    header = read_header(files[0], hdu=hdu)
    # Compute average speeds and sigma for each radial band and latitudinal bin
    avg_speeds, sigmas = compute_all_bands(acc, ycens, chw, header, nlatbins, x_kps)

    return avg_speeds, sigmas


def plot_flow_map(speeds: np.ndarray, sigmas: np.ndarray, ycen_band_rs: np.ndarray, rbands: list[int], nlatbins: int,
                  cmap: str = 'inferno'):
    """
    Plot polar maps of the radial flows.

    Parameters
    ----------
    speeds : np.ndarray
        Averaged speed over each radial band and latitudinal bin.
    sigmas : np.ndarray
        1-sigma uncertainty associated with each binned speed.
    ycen_band_rs : np.ndarray
        y-coordinates of center of bands in solar radii.
    rbands : list[int]
        Indices of the radial bands to visualize.
    nlatbins : int
        Number of angular bins over 360 degrees.
    cmap : str, optional
        Colormap for the plot (default is 'inferno').
    """
    thetas = np.linspace(0, 2 * np.pi, nlatbins + 1)

    plt.close('all')
    fig = plt.figure(figsize=(20, 8))

    vmin = speeds.min()
    vmax = speeds.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i, ridx in enumerate(rbands):

        signal = np.append(speeds[ridx], speeds[ridx][0])
        error = np.append(sigmas[ridx], sigmas[ridx][0])

        ax = fig.add_subplot(1, len(rbands), i + 1, projection='polar')
        ax.plot(thetas, signal, 'k-')
        ax.fill_between(thetas, signal - error, signal + error, alpha=0.3, color='gray')

        colors = np.array([mapper.to_rgba(v) for v in signal])
        for theta, value, err, color in zip(thetas, signal, error, colors):
            ax.plot(theta, value, 'o', color=color, ms=4)
            ax.errorbar(theta, value, yerr=err, lw=2, capsize=3, color=color)

        ax.set_title(f'Altitude = {ycen_band_rs[ridx]} Rs')
        ax.set_ylim(50, 1.05 * vmax)
        ax.set_rlabel_position(270)

    cbar_ax = fig.add_axes([0.11, 0.2, 0.8, 0.03])
    plt.colorbar(mapper, cax=cbar_ax, orientation='horizontal').ax.set_xlabel('Speed (km/s)')
    plt.savefig('Radial_Speed_Map.png')


if __name__ == "__main__":
    # Input parameters and configuration
    files = sorted(glob.glob(os.path.join(os.environ['PUNCHDATA'], '*')))
    hdu = 0  # Index of the header data unit (change if necessary)
    dt = 12  # Time offset in frames between images
    sparsity = 2  # Frame skip interval for averaging
    n_ofs = 151  # Number of spatial offsets for cross-correlation
    dr = 2  # Pixel offset increment per sample
    of0 = dt * 9  # Central offset based on expected wind speed (300 km/s)
    expected_kps_windspeed = 300  # Expected wind speed in km/s
    chw = 0.5  # Half-width of each radial band in solar radii
    nlatbins = 36  # Number of angular bins (latitude bins)
    # Define radial band centers in solar radii
    ycens = np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14])
    rbands = [0, 4, 8, 14]  # Indices of radial bands to visualize
    # End of user input

    avg_speeds, sigmas = process_corr(files, hdu, dt, sparsity, dr, n_ofs, of0, nlatbins)

    # Save the speed and sigma data in a FITS file
    data_cube = np.stack((avg_speeds, sigmas), axis=0)
    hdu = fits.PrimaryHDU(data_cube)
    hdu.writeto('speeds_sigmas.fits', overwrite=True)

    # Generate and save the radial flow map plot
    plot_flow_map(avg_speeds, sigmas, ycens, rbands, nlatbins)
