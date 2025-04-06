import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import astropy
from astropy.io import ascii

url = "https://raw.githubusercontent.com/sidchaini/LightCurveDistanceClassification/main/settings.txt"
response = requests.get(url)
assert response.status_code == 200
settings_dict = json.loads(response.text)

sns_dict = settings_dict["sns_dict"]
sns.set_theme(**sns_dict)

SCALE_FACTOR = 0.5  # scale factor for wavelength del to check wavelength range
HANDCRAFTED_WL_MIN = 999  # angstrom
HANDCRAFTED_WL_MAX = 15_000  # angstrom

# Functions to read spectra files


def ignore_equals_END_lines(filepath):
    """
    Custom reader for spectrum files that ignores lines containing '=' and lines starting with 'END'.

    This function is used as a fallback when standard astropy readers fail due to formatting issues
    in spectrum files. It filters out lines that typically cause parsing problems, such as header
    lines with key-value pairs (containing '=') and footer lines (starting with 'END').

    Parameters:
    -----------
    filepath : str
        Path to the spectrum file

    Returns:
    --------
    astropy.table.Table
        Table containing the cleaned spectrum data
    """
    with open(filepath, "r") as f:
        lines = [
            line for line in f if "=" not in line and not line.strip().startswith("END")
        ]
    table = ascii.read("\n".join(lines))
    return table


def ignore_keypair_value_lines(filepath):
    """
    Custom reader for spectrum files that removes all lines containing key:value pairs before reading it.
    However, this is only done after a check - the first two lines MUST contain JD and RA values
    (based on tests as of 2025-04-08).

    Parameters:
    -----------
    filepath : str
        Path to the spectrum file

    Returns:
    --------
    astropy.table.Table
        Table containing the cleaned spectrum data
    """
    with open(filepath, "r") as f:
        all_lines = f.readlines()

    # Check if first two lines match the required format
    if (
        len(all_lines) >= 2
        and all_lines[0].strip().startswith("JD:")
        and all_lines[1].strip().startswith("RA:")
    ):
        lines = [line for line in all_lines if ":" not in line]
    else:
        lines = all_lines

    table = ascii.read("\n".join(lines))
    return table


def read_spectra(filepath):
    """
    Read a spectrum file and return a pandas DataFrame.

    This function attempts to read various spectrum file formats commonly used in
    astronomical data. It handles different file structures including those with
    headers, comments, and various column arrangements.

    Parameters:
    -----------
    filepath : str
        Path to the spectrum file

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the spectrum data with columns for wavelength, flux,
        and possibly error values

    Notes:
    ------
    If the standard astropy reader fails, it falls back to a custom reader that
    ignores lines containing '=' and lines starting with 'END', which can cause
    parsing issues in some spectrum formats.
    """
    try:
        df = ascii.read(filepath).to_pandas()
    except ascii.InconsistentTableError:
        df = ignore_equals_END_lines(filepath).to_pandas()
    return df


# Functions for checks on spectra read


def check_wavelength_increasing(wavelengths):
    """
    Check if wavelength values are monotonically increasing.

    Parameters:
    -----------
    wavelengths : numpy.ndarray
        1D array of wavelength values

    Returns:
    --------
    bool
        True if wavelengths are monotonically increasing, False otherwise
    """
    is_increasing = np.all(np.diff(wavelengths) > 0)
    return is_increasing


def check_flux_positive(flux, frac_positive=0.8):
    """
    Check if a sufficient fraction of flux values are positive.

    Parameters:
    -----------
    flux : numpy.ndarray
        1D array of flux values
    frac_positive : float, default=0.8
        Minimum fraction of positive flux values required

    Returns:
    --------
    bool
        True if the fraction of positive flux values exceeds the threshold,
        False otherwise
    """
    is_positive = np.sum(flux > 0) / len(flux) > frac_positive
    return is_positive


def check_df(df, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda):
    # Check no. of readings make sense
    n_samples = int((lambda_max - lambda_min) // del_lambda)
    if df.shape[0] not in [n_samples, n_samples + 1]:
        raise ValueError(
            f"Expected {n_samples} rows based on wavelength range, but got {df.shape[0]}"
        )

    if df.shape[1] in [2, 3, 4]:
        wavelength, flux = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    else:
        raise NotImplementedError(
            f"Processing for dataframes with {df.shape[1]} columns is not supported yet"
        )
    check_wavelengthflux(
        wavelength, flux, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda
    )


def check_wavelengthflux(
    wavelength, flux, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda
):
    # Check wavelength range is correct
    if abs(wavelength[0] - lambda_min) > SCALE_FACTOR * del_lambda:
        raise ValueError(
            f"First wavelength {wavelength[0]} doesn't match expected lambda_min {lambda_min}"
        )

    if abs(wavelength[-1] - lambda_max) > SCALE_FACTOR * del_lambda:
        raise ValueError(
            f"Last wavelength {wavelength[-1]} doesn't match expected lambda_max {lambda_max}"
        )

    # Check wavelength is monotonically increasing
    if not check_wavelength_increasing(wavelength):
        raise ValueError("Wavelengths are not monotonically increasing")

    # Check wavelength units are in angstrom and if not, convert
    if wl_unit != "Angstrom":
        if wl_unit == "Micrometre":
            wavelength = wavelength * 10_000  # convert to angstrom
        elif wl_unit == "nm":  # Fixed syntax error
            wavelength = wavelength * 10  # convert to angstrom
        else:
            raise NotImplementedError(
                f"Wavelength unit conversion from '{wl_unit}' to Angstrom not implemented"
            )

    # Check wavelength is in a range that make sense
    if wavelength[0] < HANDCRAFTED_WL_MIN:
        raise ValueError(
            f"First wavelength {wavelength[0]} is below minimum allowed value {HANDCRAFTED_WL_MIN}"
        )
    if wavelength[-1] > HANDCRAFTED_WL_MAX:
        raise ValueError(
            f"Last wavelength {wavelength[-1]} exceeds maximum allowed value {HANDCRAFTED_WL_MAX}"
        )

    # Check flux units and convert
    if spec_unit != "erg cm(-2) sec(-1) Ang(-1)":
        # if spec_unit=="mJy":
        #     flux = ###
        # else:
        # raise NotImplementedError # add message and mention spec_unit
        raise NotImplementedError(
            f"Spectral unit conversion from '{spec_unit}' not implemented"
        )  # add message and mention spec_unit

    # Check if flux are all positive
    if not check_flux_positive(flux):
        raise AttributeError("Flux is not positive!")


# LIMIT TO FEW DAYS BEFORE AND MAXIMUM OF SPECTRA - FOR NOW
# def ####
