import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import astropy
import os
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

    Tries multiple reader functions in sequence until one succeeds.

    Parameters:
    -----------
    filepath : str
        Path to the spectrum file

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the spectrum data

    Notes:
    ------
    If the standard astropy reader fails, it falls back to a custom reader that
    ignores lines containing '=' and lines starting with 'END', which can cause
    parsing issues in some spectrum formats.

    Raises:
    -------
    ValueError
        If all reader functions fail

    """
    # List of reader functions to try in order
    reader_functions = [ascii.read, ignore_equals_END_lines, ignore_keypair_value_lines]

    errors = []

    for reader_func in reader_functions:
        try:
            df = reader_func(filepath).to_pandas()
            return df
        except Exception as e:
            errors.append(f"{reader_func.__name__}: {str(e)}")
            continue

    # If we get here, all readers failed
    error_msg = "All spectrum readers failed:\n" + "\n".join(errors)
    raise ValueError(error_msg)


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


def check_flux_units_erg_cm2_s_ang(flux, frac_positive_threshold=0.75):
    """
    Checks if unlabeled flux values are plausibly in units of
    'erg cm(-2) sec(-1) Ang(-1)' based on a magnitude range
    and basic signal properties.

    Astronomical sources commonly observed in optical/NIR spectra have flux densities
    in these units typically falling between roughly 1e-18 and 1e-10.
    This check verifies if the median absolute flux falls within this plausible range.
    It also requires a minimum fraction of flux values to be positive, helping
    to distinguish signal from pure noise centered on zero.

    Parameters:
    -----------
    flux : numpy.ndarray
        Flux values to check. Assumed to be cleaned of NaNs if necessary.
    frac_positive_threshold : float, optional
        Minimum fraction of positive flux values required.
    Returns:
    --------
    bool
        True if flux values fall within the expected range and meet the
        positive fraction criterion, suggesting plausible units, False otherwise.

    Notes:
    ------
    This is a heuristic check based on typical values. It cannot definitively
    confirm the units but serves as a useful filter against grossly miscalibrated
    data or data in vastly different unit systems (e.g., counts, relative flux).
    Requires wavelength checks to be performed separately.
    """
    if flux is None or len(flux) == 0:
        return False  # Cannot check empty flux array

    # Check 1: Typical magnitude range for erg cm-2 s-1 Å-1
    # Use nanmedian to be robust against potential NaNs
    median_abs_flux = np.nanmedian(np.abs(flux))
    # Check if median_abs_flux is NaN (can happen if all flux values are NaN)
    if np.isnan(median_abs_flux):
        return False

    is_in_range = (median_abs_flux > 1e-18) and (median_abs_flux < 1e-10)

    # Check 2: Ensure it's not just noise around zero (sufficient positive values)
    # Use nanmean to ignore NaNs when calculating the fraction
    positive_fraction = np.nanmean(flux > 0)
    # Check if positive_fraction is NaN (can happen if all flux values are NaN)
    if np.isnan(positive_fraction):
        return False

    is_sufficiently_positive = positive_fraction > frac_positive_threshold

    return is_in_range and is_sufficiently_positive


def check_spectrafile(df, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda):
    """
    Validates a spectrum dataframe against expected parameters and extracts wavelength and flux data.

    This function performs several validation checks on the input spectrum dataframe:
    1. Verifies the number of rows matches the expected count based on wavelength range and step size (Error #1)
    2. Checks that the dataframe has a supported number of columns (2, 3, or 4) (Error #2)
    3. Extracts wavelength and flux data from the first two columns
    4. Delegates to check_wavelengthflux() for additional validation of:
       - 4.1. Wavelength range boundaries (Errors #3, #4)
       - 4.2. Monotonically increasing wavelengths (Error #5)
       - 4.3. Wavelength unit conversion if needed (Error #6)
       - 4.4. Wavelength values within allowed physical limits (Errors #7, #8)
       - 4.5. Flux units validation (Errors #9, #10)
       - 4.6. Flux mostly positive (Error #11)

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing spectrum data with wavelength in first column and flux in second
    wl_unit : str
        Unit of wavelength values (e.g., 'Angstrom', 'Micrometre', 'nm')
    spec_unit : str
        Unit of flux values (e.g., 'erg cm(-2) sec(-1) Ang(-1)', 'Other')
    lambda_min : float
        Expected minimum wavelength value
    lambda_max : float
        Expected maximum wavelength value
    del_lambda : float
        Expected wavelength step size

    Returns:
    --------
    None
        Function raises ValueError or NotImplementedError if validation fails

    Raises:
    -------
    ValueError
        If number of rows doesn't match expected count (Error #1) or if data validation in check_wavelengthflux fails (Errors #3, #4, #5, #7, #8, #11)
    NotImplementedError
        If dataframe has unsupported number of columns (Error #2) or if unit conversions are not implemented (Errors #6, #9, #10)
    """
    # 1. Check if number of rows matches expected count based on wavelength range
    n_samples = int((lambda_max - lambda_min) // del_lambda)
    if df.shape[0] not in [n_samples, n_samples + 1]:
        raise ValueError(
            f"Error #1 (check_spectrafile): Expected {n_samples} or {n_samples + 1} rows based on wavelength range, but got {df.shape[0]}"
        )

    # 2. Check if dataframe has a supported number of columns (2, 3, or 4)
    if df.shape[1] in [2, 3, 4]:
        # 3. Extract wavelength and flux data from the first two columns
        wavelength, flux = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    else:
        raise NotImplementedError(
            f"Error #2 (check_spectrafile): Processing for dataframes with {df.shape[1]} columns is not supported yet"
        )

    # 4. Delegate to check_wavelengthflux() for additional validation
    # 4.1. Check wavelength range boundaries
    # 4.2. Check wavelength is monotonically increasing
    # 4.3. Check and convert wavelength units if needed
    # 4.4. Verify wavelength values are within allowed physical limits
    # 4.5. Validate flux units
    # 4.6. Check if flux is mostly positive
    check_wavelengthflux(
        wavelength, flux, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda
    )


def check_wavelengthflux(
    wavelength, flux, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda
):
    """
    Performs detailed validation checks on wavelength and flux arrays extracted from a spectrum.

    Called by check_spectrafile().

    Checks performed:
    1. Wavelength range matches expected min/max within tolerance (Errors #3, #4)
    2. Wavelength values are monotonically increasing (Error #5)
    3. Wavelength units are 'Angstrom' or convertible ('Micrometre', 'nm') (Error #6)
    4. Wavelength values are within physically plausible limits (HANDCRAFTED_WL_MIN/MAX) (Errors #7, #8)
    5. Flux units are 'erg cm(-2) sec(-1) Ang(-1)' or 'Other' (with validation) (Errors #9, #10)
    6. Flux values are sufficiently positive (Error #11)

    Parameters:
    -----------
    wavelength : numpy.ndarray
        1D array of wavelength values
    flux : numpy.ndarray
        1D array of flux values
    wl_unit : str
        Unit of wavelength values (e.g., 'Angstrom', 'Micrometre', 'nm')
    spec_unit : str
        Unit of flux values (e.g., 'erg cm(-2) sec(-1) Ang(-1)', 'Other')
    lambda_min : float
        Expected minimum wavelength value
    lambda_max : float
        Expected maximum wavelength value
    del_lambda : float
        Expected wavelength step size

    Returns:
    --------
    None
        Function raises ValueError or NotImplementedError if validation fails

    Raises:
    -------
    ValueError
        If wavelength range is incorrect (Errors #3, #4)
        If wavelengths are not monotonic (Error #5)
        If wavelengths are outside physical limits (Errors #7, #8)
        If flux is not sufficiently positive (Error #11)
        If wavelength or flux contain NaN or Inf values (Errors #12, #13, #14, #15)
    NotImplementedError
        If wavelength unit conversion is not supported (Error #6)
        If flux units are 'Other' but invalid, or conversion not supported (Errors #9, #10)
    """
    # 0. Check for NaN/Inf values
    if np.isnan(wavelength).any():
        raise ValueError(
            "Error #12 (check_wavelengthflux): Wavelength array contains NaN values"
        )
    if np.isinf(wavelength).any():
        raise ValueError(
            "Error #13 (check_wavelengthflux): Wavelength array contains Inf values"
        )
    if np.isnan(flux).any():
        raise ValueError(
            "Error #14 (check_wavelengthflux): Flux array contains NaN values"
        )
    if np.isinf(flux).any():
        raise ValueError(
            "Error #15 (check_wavelengthflux): Flux array contains Inf values"
        )

    # 1. Check if wavelength range is correct
    if abs(wavelength[0] - lambda_min) > SCALE_FACTOR * del_lambda:
        raise ValueError(
            f"Error #3 (check_wavelengthflux): First wavelength {wavelength[0]} doesn't match expected lambda_min {lambda_min} within tolerance {SCALE_FACTOR * del_lambda}"
        )

    if abs(wavelength[-1] - lambda_max) > SCALE_FACTOR * del_lambda:
        raise ValueError(
            f"Error #4 (check_wavelengthflux): Last wavelength {wavelength[-1]} doesn't match expected lambda_max {lambda_max} within tolerance {SCALE_FACTOR * del_lambda}"
        )

    # 2. Check if wavelength is monotonically increasing
    if not check_wavelength_increasing(wavelength):
        raise ValueError(
            "Error #5 (check_wavelengthflux): Wavelengths are not monotonically increasing"
        )

    # 3. Check wavelength units are in angstrom and if not, convert
    if wl_unit != "Angstrom":
        if wl_unit == "Micrometre":
            wavelength = wavelength * 10_000  # convert to angstrom
        elif wl_unit == "nm":  # Fixed syntax error
            wavelength = wavelength * 10  # convert to angstrom
        else:
            raise NotImplementedError(
                f"Error #6 (check_wavelengthflux): Wavelength unit conversion from '{wl_unit}' to Angstrom not implemented"
            )

    # 4. Check wavelength is in a range that make sense
    if wavelength[0] < HANDCRAFTED_WL_MIN:
        raise ValueError(
            f"Error #7 (check_wavelengthflux): First wavelength {wavelength[0]} is below minimum allowed value {HANDCRAFTED_WL_MIN}"
        )
    if wavelength[-1] > HANDCRAFTED_WL_MAX:
        raise ValueError(
            f"Error #8 (check_wavelengthflux): Last wavelength {wavelength[-1]} exceeds maximum allowed value {HANDCRAFTED_WL_MAX}"
        )

    # 5. Check flux units and convert

    if spec_unit == "erg cm(-2) sec(-1) Ang(-1)":
        pass
    elif spec_unit == "Other":
        if not check_flux_units_erg_cm2_s_ang(flux):
            raise NotImplementedError(
                f"Error #9 (check_wavelengthflux): Flux units are 'Other' but do not resemble expected erg cm(-2) sec(-1) Ang(-1) values."
            )
    else:
        # if spec_unit=="mJy":
        #     flux = ###
        # else:
        # raise NotImplementedError # add message and mention spec_unit
        raise NotImplementedError(
            f"Error #10 (check_wavelengthflux): Spectral unit conversion from '{spec_unit}' not implemented"
        )  # add message and mention spec_unit

    # 6. Check if flux are all positive
    if not check_flux_positive(flux):
        raise ValueError(
            "Error #11 (check_wavelengthflux): Flux is not sufficiently positive!"
        )


def check_and_plot_spectra(
    spectra_fn,
    meta_df,
    spectra_dir="../1. download ALL wise data/wiserep_data/spectra/",
    plot=True,
):
    """
    Wrapper function that loads a spectrum file, performs validation checks, and optionally plots it.

    """
    if meta_df.index.name != "Ascii file":
        meta_df.set_index("Ascii file")
        print("Setting ascii file as index.")
    spectra_meta = meta_df.loc[spectra_fn]
    spectra_df = read_spectra(os.path.join(spectra_dir, spectra_fn))

    wl_unit = spectra_meta["WL Units"]
    spec_unit = spectra_meta["Spec. units"]
    flux_ucoeff = spectra_meta["Flux Unit Coefficient"]
    lambda_min = spectra_meta["Lambda-min"]
    lambda_max = spectra_meta["Lambda-max"]
    del_lambda = spectra_meta["Del-Lambda"]

    # Perform validation checks
    check_spectrafile(
        spectra_df, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda
    )

    # Plot the spectrum if requested
    if plot:
        import matplotlib.pyplot as plt

        x = spectra_df.to_numpy()[:, 0]  # wavelength
        y = spectra_df.to_numpy()[:, 1]  # flux
        plt.figure(figsize=(10, 6))
        plt.plot(x, y)
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Flux (erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$)")
        plt.title(f"Spectrum: {os.path.basename(spectra_fn)}")
        plt.grid(True, alpha=0.3)
        plt.show()


# LIMIT TO FEW DAYS BEFORE AND MAXIMUM OF SPECTRA - FOR NOW
# def ####
