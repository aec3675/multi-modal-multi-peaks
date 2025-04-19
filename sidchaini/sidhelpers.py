"""
Helper functions for astronomical spectra analysis, including reading, validation, and plotting.

Error Numbering Convention:
Errors raised by functions in this module follow an 'X.Y' format:
'Error X.Y (function_name): Message'

Where 'X' denotes the logical task category and 'Y' is the specific error within that category.
This provides traceability and allows related errors to be grouped.

Category Summary:
1.x: _find_spectral_columns - Column identification failures during positional fallback.
2.x: read_spectra - File reading process errors (e.g., reader type, all readers fail).
3.x: check_spectrafile - Initial parameter checks (e.g., Del-Lambda).
4.x: check_spectrafile - Row count validation.
5.x: check_spectrafile - Handling re-raised errors from column identification.
6.x: check_spectrafile - Data extraction/conversion failures (e.g., KeyError, TypeError).
7.x: check_wavelengthflux - Basic data integrity checks (NaN, Inf).
8.x: check_wavelengthflux - Wavelength unit conversion issues.
9.x: check_wavelengthflux - Wavelength validation against metadata (range, monotonicity).
10.x: check_wavelengthflux - Wavelength validation against physical plausibility limits.
11.x: check_wavelengthflux - Flux unit handling/validation (plausibility, known types).
12.x: check_wavelengthflux - Flux value checks (positivity).
13.x: check_and_plot_spectra - Metadata lookup errors.
14.x: check_and_plot_spectra - Data conversion errors during plotting preparation.
"""

# --- Standard Library Imports ---
import os
import json
import warnings

# --- Third-Party Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import astropy
from astropy.io import ascii

# --- Configuration & Global Settings ---
# Fetch settings from remote URL (e.g., plotting styles)
try:
    _SETTINGS_URL = "https://raw.githubusercontent.com/sidchaini/LightCurveDistanceClassification/main/settings.txt"
    _response = requests.get(_SETTINGS_URL)
    _response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    _settings_dict = json.loads(_response.text)
    _sns_dict = _settings_dict.get("sns_dict", {})  # Use .get for safety
    sns.set_theme(**_sns_dict)
except requests.exceptions.RequestException as e:
    warnings.warn(
        f"Could not fetch settings from {_SETTINGS_URL}. Using default plotting styles. Error: {e}",
        UserWarning,
    )
except json.JSONDecodeError as e:
    warnings.warn(
        f"Could not decode settings JSON from {_SETTINGS_URL}. Using default plotting styles. Error: {e}",
        UserWarning,
    )
except Exception as e:  # Catch any other unexpected errors during setup
    warnings.warn(
        f"An unexpected error occurred during settings setup: {e}. Using default plotting styles.",
        UserWarning,
    )


# --- Physics/Astro Constants ---

# Wavelength Validation
SCALE_FACTOR = 0.5  # Tolerance factor for wavelength range checks (relative to spectral resolution del_lambda)
HANDCRAFTED_WL_MIN = 999  # Minimum plausible optical/NIR wavelength observed (Angstrom)
HANDCRAFTED_WL_MAX = (
    15_000  # Maximum plausible optical/NIR wavelength observed (Angstrom)
)

# Flux Validation
MIN_POSITIVE_FLUX_FRACTION = (
    0.80  # Required fraction of positive flux points for general validation
)
MIN_POSITIVE_FLUX_FRACTION_UNIT_CHECK = (
    0.75  # Required fraction for flux unit plausibility check
)
FLUX_UNIT_CHECK_MIN_MAG = 1e-18  # Plausible minimum flux magnitude (erg cm-2 s-1 Å-1)
FLUX_UNIT_CHECK_MAX_MAG = 1e-10  # Plausible maximum flux magnitude (erg cm-2 s-1 Å-1)


# --- Column Name Identification ---
# Define common column names (lowercase) for robust identification across various file formats.
# The lists are ordered by likely prevalence or specificity.

WAVELENGTH_COLUMN_NAMES = [
    name.lower()
    for name in [
        "wavelength",
        "#wavelength",
        "wavelen",
        "wavelength_angstrom",
        "lambda_aa",
        "wavelength(aa)",
        "wavelength(angstroms)",
        "wavelength (å)",
        "wl[å]",
        "wl[\aa]",
        "wave",
        "wav",
        "wl",
        "lambda",
        "lam",
        "wave-obs",
        "waveobs",
        "obswave",
        "air_wave",
        "vacuum_wave",
        "wavelength(um)",
        "angstrom	10^-18",
        "wl(a)",
        "aa",
    ]
]

FLUX_COLUMN_NAMES = [
    name.lower()
    for name in [
        "flux",
        "flux_density",
        "flam",
        "flambda",
        "f_lam",
        "intensity",
        "flux(flam)",
        "flux(mjy)",
        "flux (erg/s/cm2/å)",
        "flux[μjy]",
        "flux(10-15)",
        "normalized_flux",
        "erg/s/cm^2/angstrom",
        "absflux",
        "flux_cgs_aa",
        "flux_tell_corrected",
        "flux_not_tell_corrected",
        "flux_smoothed",
        "spec_sum",
        "spec_optimal",
        "response",
        "absflux(10^-16ergcm^-2s^-1(aa)^-1)",
        "absflux(10^-14ergcm^-2s^-1(aa)^-1)",
        "flux_egs",
    ]
]


# ==============================================================================
# ==                          CORE HELPER FUNCTIONS                           ==
# ==============================================================================


def _find_spectral_columns(df):
    """
    Identify wavelength and flux columns in a spectrum DataFrame.

    Tries to match column names against predefined lists (case-insensitive).
    If names aren't found, it attempts to use the first two columns (index 0 for
    wavelength, index 1 for flux) as a fallback, issuing a warning.

    This heuristic approach is needed due to the variety of naming conventions
    found in astronomical data files.

    Raises specific ValueErrors (Error 1.x) if columns cannot be uniquely
    identified by name or the positional fallback is not possible (e.g.,
    only one column exists).
    """
    original_columns = df.columns
    lower_columns = [str(col).lower() for col in original_columns]
    wavelength_col_name = None
    flux_col_name = None

    # --- Step 1: Attempt to find wavelength column by name ---
    for name in WAVELENGTH_COLUMN_NAMES:
        try:
            idx = lower_columns.index(name)
            wavelength_col_name = original_columns[idx]
            break  # Found a match, stop searching
        except ValueError:
            continue  # Name not in list, try next one

    # --- Step 2: Attempt to find flux column by name ---
    for name in FLUX_COLUMN_NAMES:
        try:
            idx = lower_columns.index(name)
            # Crucially, ensure we didn't identify the *same* column as wavelength
            if original_columns[idx] != wavelength_col_name:
                flux_col_name = original_columns[idx]
                break  # Found a different match, stop searching
            # If it IS the same column, continue searching other flux names
        except ValueError:
            continue  # Name not in list, try next one

    # --- Step 3: Handle identification results and apply fallbacks ---
    if wavelength_col_name and flux_col_name:
        # Best case: Successfully identified both by name
        return wavelength_col_name, flux_col_name

    elif wavelength_col_name and not flux_col_name:
        # Found wavelength by name, attempt positional fallback for flux
        if df.shape[1] > 1:
            wl_col_idx = df.columns.get_loc(wavelength_col_name)
            # Try the 'other' column (index 0 or 1)
            flux_col_idx = 1 if wl_col_idx == 0 else 0
            if flux_col_idx < len(original_columns):  # Check index is valid
                flux_col_name = original_columns[flux_col_idx]
                warnings.warn(
                    f"Wavelength column '{wavelength_col_name}' found by name, but flux column not found. "
                    f"Falling back to positional column '{flux_col_name}' (index {flux_col_idx}) for flux.",
                    UserWarning,
                )
                return wavelength_col_name, flux_col_name
            else:
                # This case is rare unless df has only 1 column which was identified as wavelength
                raise ValueError(
                    f"Error 1.1 (_find_spectral_columns): Found wavelength '{wavelength_col_name}', "
                    f"but cannot determine flux column positionally (invalid index {flux_col_idx})."
                )
        else:  # Only one column total
            raise ValueError(
                f"Error 1.2 (_find_spectral_columns): Found wavelength column '{wavelength_col_name}' by name, "
                f"but cannot determine flux column positionally (only {df.shape[1]} column)."
            )

    elif not wavelength_col_name and flux_col_name:
        # Found flux by name, attempt positional fallback for wavelength
        if df.shape[1] > 1:
            flux_col_idx = df.columns.get_loc(flux_col_name)
            # Try the 'other' column (index 0 or 1)
            wl_col_idx = 0 if flux_col_idx == 1 else 1
            if wl_col_idx < len(original_columns):  # Check index is valid
                wavelength_col_name = original_columns[wl_col_idx]
                warnings.warn(
                    f"Flux column '{flux_col_name}' found by name, but wavelength column not found. "
                    f"Falling back to positional column '{wavelength_col_name}' (index {wl_col_idx}) for wavelength.",
                    UserWarning,
                )
                return wavelength_col_name, flux_col_name
            else:
                # This case is rare unless df has only 1 column which was identified as flux
                raise ValueError(
                    f"Error 1.3 (_find_spectral_columns): Found flux '{flux_col_name}', "
                    f"but cannot determine wavelength column positionally (invalid index {wl_col_idx})."
                )
        else:  # Only one column total
            raise ValueError(
                f"Error 1.4 (_find_spectral_columns): Found flux column '{flux_col_name}' by name, "
                f"but cannot determine wavelength column positionally (only {df.shape[1]} column)."
            )

    else:
        # --- Step 4: Full Positional Fallback (Neither found by name) ---
        if df.shape[1] >= 2:
            wavelength_col_name = original_columns[0]
            flux_col_name = original_columns[1]
            warnings.warn(
                f"Could not identify wavelength or flux columns by name. "
                f"Falling back to positional columns: index 0 ('{wavelength_col_name}') for wavelength, "
                f"index 1 ('{flux_col_name}') for flux.",
                UserWarning,
            )
            return wavelength_col_name, flux_col_name
        else:  # Less than 2 columns, cannot fallback
            raise ValueError(
                f"Error 1.5 (_find_spectral_columns): Could not identify wavelength/flux columns by name, "
                f"and only {df.shape[1]} column(s) available for positional fallback."
            )


# ==============================================================================
# ==                         SPECTRA FILE READERS                             ==
# ==============================================================================

# --- Custom Reader Functions (for problematic file formats) ---


def _ignore_equals_END_lines(filepath):
    """
    Custom reader: Ignores lines containing '=' or starting with 'END'.

    Handles common non-standard formats where header/footer lines break standard parsers.
    """
    try:
        with open(filepath, "r") as f:
            lines = [
                line
                for line in f
                if "=" not in line and not line.strip().upper().startswith("END")
            ]
        # Use astropy.ascii for robust table parsing after filtering
        table = ascii.read("\n".join(lines))
        return table
    except Exception as e:
        # Add context to the error before it's caught by read_spectra
        raise type(e)(f"Error in _ignore_equals_END_lines for {filepath}: {e}") from e


def _ignore_keypair_value_lines(filepath):
    """
    Custom reader: Removes lines with 'key: value' pairs if header matches pattern.

    Specifically targets a format observed where the first two lines are 'JD:' and 'RA:',
    followed by data, potentially interrupted by other 'key: value' lines.
    """
    try:
        with open(filepath, "r") as f:
            all_lines = f.readlines()

        # Check if first two lines match the expected header pattern
        if (
            len(all_lines) >= 2
            and all_lines[0].strip().upper().startswith("JD:")
            and all_lines[1].strip().upper().startswith("RA:")
        ):
            # Filter out any line containing a colon if pattern matches
            lines = [line for line in all_lines if ":" not in line]
        else:
            # If pattern doesn't match, process all lines (standard astropy might handle it)
            lines = all_lines

        table = ascii.read("\n".join(lines))
        return table
    except Exception as e:
        # Add context to the error
        raise type(e)(
            f"Error in _ignore_keypair_value_lines for {filepath}: {e}"
        ) from e


# --- Main Spectrum Reading Function ---


def read_spectra(filepath):
    """
    Read a spectrum file using multiple strategies and return a pandas DataFrame.

    Attempts reading with standard `astropy.io.ascii.read` first. If that fails,
    it tries custom reader functions (`_ignore_equals_END_lines`, `_ignore_keypair_value_lines`)
    designed to handle specific non-standard formatting issues observed in real data.

    Raises:
        ValueError (Error 2.1): If a reader returns an unexpected data type (not astropy Table or pandas DataFrame).
        ValueError (Error 2.2): If all attempted reading methods fail, summarizing the errors from each attempt.
        FileNotFoundError: If the input `filepath` does not exist.
        Other exceptions from underlying readers (e.g., `astropy.io.ascii.InconsistentTableError`).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Spectrum file not found: {filepath}")

    # List of reader functions to try, in order of preference
    # Standard reader first, then custom workarounds
    reader_functions = [
        ascii.read,
        _ignore_equals_END_lines,
        _ignore_keypair_value_lines,
    ]

    errors = []  # Keep track of errors from each failed reader

    for reader_func in reader_functions:
        try:
            table = reader_func(filepath)

            # Ensure the output is a pandas DataFrame
            if isinstance(table, astropy.table.Table):
                df = table.to_pandas()
            elif isinstance(table, pd.DataFrame):
                df = table  # Already a DataFrame
            else:
                # This case should be rare with the current readers
                raise ValueError(
                    f"Error 2.1 (read_spectra): Reader {reader_func.__name__} returned unexpected type {type(table)} for file {filepath}"
                )

            # Success! Return the DataFrame
            return df

        except Exception as e:
            # Record the error and try the next reader
            errors.append(f"  - {reader_func.__name__}: {type(e).__name__}({str(e)})")
            continue

    # If we reach here, all readers failed
    error_summary = "\n".join(errors)
    raise ValueError(
        f"Error 2.2 (read_spectra): All spectrum readers failed for file '{os.path.basename(filepath)}'. Errors encountered:\n{error_summary}"
    )


# ==============================================================================
# ==                      DATA VALIDATION FUNCTIONS                           ==
# ==============================================================================

# --- Basic Array Checks ---


def check_wavelength_increasing(wavelengths):
    """
    Check if wavelength values are strictly monotonically increasing.
    Handles single-point spectra correctly.
    """
    # Edge case: A single point is trivially monotonic
    if len(wavelengths) <= 1:
        return True
    # Check if all differences are positive
    return np.all(np.diff(wavelengths) > 0)


def check_flux_positive(flux, frac_positive=MIN_POSITIVE_FLUX_FRACTION):
    """
    Check if a sufficient fraction of flux values are positive.

    Helps distinguish real signal from noise centered around zero, or spectra
    with excessive negative regions (e.g., due to over-subtraction). Ignores NaNs.
    """
    if flux is None or len(flux) == 0:
        return False  # Cannot check empty array

    # Calculate fraction of positive points, ignoring NaNs
    with warnings.catch_warnings():  # Suppress RuntimeWarning for all-NaN slices
        warnings.simplefilter("ignore", category=RuntimeWarning)
        positive_fraction = np.nanmean(flux > 0)

    if np.isnan(positive_fraction):  # Handle case where all flux values are NaN
        return False

    return positive_fraction > frac_positive


def check_flux_units_erg_cm2_s_ang(
    flux, frac_positive_threshold=MIN_POSITIVE_FLUX_FRACTION_UNIT_CHECK
):
    """
    Check if flux values are plausibly in 'erg cm-2 s-1 Å-1'.

    Uses heuristic checks based on typical astronomical flux magnitudes in these
    units and requires a minimum fraction of positive flux values. Ignores NaNs.

    This is useful when metadata is missing or ambiguous ('Other').
    """
    if flux is None or len(flux) == 0:
        return False

    # --- Check 1: Plausible Magnitude Range ---
    # Use nanmedian for robustness against outliers and NaNs
    median_abs_flux = np.nanmedian(np.abs(flux))
    if np.isnan(median_abs_flux):  # Handle case where all flux values are NaN
        return False
    is_in_range = FLUX_UNIT_CHECK_MIN_MAG < median_abs_flux < FLUX_UNIT_CHECK_MAX_MAG

    # --- Check 2: Sufficient Positive Signal ---
    # Check if enough points are positive (using the dedicated function)
    is_sufficiently_positive = check_flux_positive(
        flux, frac_positive=frac_positive_threshold
    )

    return is_in_range and is_sufficiently_positive


# --- Comprehensive Spectrum Validation ---


def check_spectrafile(df, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda):
    """
    Validate spectrum DataFrame content against metadata and physical expectations.

    Performs checks on row count, identifies columns, extracts data, and delegates
    detailed array checks to `check_wavelengthflux`.

    Raises specific ValueErrors or NotImplementedErrors for failures.
    Returns identified wavelength and flux column names on success.
    """
    # --- Check 1: Validate Input Parameters ---
    if del_lambda <= 0:
        raise ValueError(
            "Error 3.1 (check_spectrafile): Del-Lambda (spectral resolution) must be positive."
        )

    # --- Check 2: Validate Row Count ---
    # Expected number of data points based on wavelength range and resolution
    # Using np.floor provides robustness against minor floating point inaccuracies
    n_samples_expected = int(np.floor((lambda_max - lambda_min) / del_lambda))
    # Allow for off-by-one differences due to endpoint inclusion/exclusion or rounding
    expected_rows_low = max(0, n_samples_expected)  # Ensure lower bound is non-negative
    expected_rows_high = n_samples_expected + 1
    if not (expected_rows_low <= df.shape[0] <= expected_rows_high):
        raise ValueError(
            f"Error 4.1 (check_spectrafile): Unexpected number of rows. "
            f"Expected ~{n_samples_expected} (range {expected_rows_low}-{expected_rows_high}) based on "
            f"λ_min={lambda_min}, λ_max={lambda_max}, Δλ={del_lambda}. Found {df.shape[0]} rows."
        )

    # --- Check 3: Identify Wavelength and Flux Columns ---
    try:
        wavelength_col_name, flux_col_name = _find_spectral_columns(df)
    except ValueError as e:
        # Re-raise the specific error (1.x) from the helper function, adding context
        raise ValueError(
            f"Error 5.1 (check_spectrafile): Failed to identify spectral columns. Original error: {e}"
        ) from e  # Preserve original traceback

    # --- Check 4: Extract and Convert Data ---
    try:
        # Attempt conversion to float numpy arrays immediately for early type checking
        wavelength = df[wavelength_col_name].to_numpy(dtype=float)
        flux = df[flux_col_name].to_numpy(dtype=float)
    except KeyError as e:
        # Should be rare if _find_spectral_columns worked, but catch defensively
        raise ValueError(
            f"Error 6.1 (check_spectrafile): Identified column name '{e}' not found in DataFrame index {df.columns}."
        )
    except (ValueError, TypeError) as e:
        # Catch errors if data in columns cannot be cast to float
        raise ValueError(
            f"Error 6.2 (check_spectrafile): Failed to convert column data to numeric type. "
            f"Check columns '{wavelength_col_name}' and '{flux_col_name}' for non-numeric values. Original error: {type(e).__name__}({e})"
        )

    # --- Check 5: Perform Detailed Wavelength and Flux Array Validation ---
    # This function performs checks for NaNs, units, range, monotonicity etc.
    # It raises specific errors (7.x - 12.x) upon failure.
    check_wavelengthflux(
        wavelength, flux, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda
    )

    # --- Success ---
    # Return the validated column names for potential use (e.g., plotting)
    return wavelength_col_name, flux_col_name


def check_wavelengthflux(
    wavelength, flux, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda
):
    """
    Perform detailed checks on extracted wavelength and flux numpy arrays.

    Validates against NaN/Inf values, handles wavelength unit conversion (if needed),
    checks wavelength range and monotonicity, assesses physical plausibility,
    validates flux units (including heuristic checks for 'Other'), and checks
    for sufficient positive flux signal.

    Called internally by `check_spectrafile`. Raises specific errors (7.x - 12.x).
    """
    # --- Check 0: Basic Data Integrity (NaN/Inf) ---
    # These checks are crucial before performing numerical operations.
    if np.isnan(wavelength).any():
        raise ValueError(
            "Error 7.1 (check_wavelengthflux): Wavelength array contains NaN values."
        )
    if np.isinf(wavelength).any():
        raise ValueError(
            "Error 7.2 (check_wavelengthflux): Wavelength array contains Inf values."
        )
    if np.isnan(flux).any():
        # Note: Depending on analysis, some NaNs in flux might be acceptable.
        # For now, we treat them as errors to ensure clean data for basic checks.
        # Consider refining this if specific NaN handling is required later.
        # warnings.warn(
        #     "Flux array contains NaN values. Treating as error for now.", UserWarning
        # )
        raise ValueError(
            "Error 7.3 (check_wavelengthflux): Flux array contains NaN values."
        )
    if np.isinf(flux).any():
        raise ValueError(
            "Error 7.4 (check_wavelengthflux): Flux array contains Inf values."
        )

    # --- Check 1: Wavelength Unit Conversion (if necessary) ---
    # Ensure wavelength is in Angstroms for consistent range checks.
    # Use a copy to avoid modifying the original array passed to the function.
    wl_angstrom = np.copy(wavelength)
    if wl_unit != "Angstrom":
        if wl_unit == "Micrometre" or wl_unit == "um":
            wl_angstrom *= 10_000  # Convert microns to Angstroms
            # warnings.warn(
            #     f"Converted wavelength from {wl_unit} to Angstrom.", UserWarning
            # )
        elif wl_unit == "nm":
            wl_angstrom *= 10  # Convert nm to Angstroms
            # warnings.warn(
            #     f"Converted wavelength from {wl_unit} to Angstrom.", UserWarning
            # )
        else:
            # If unit is unknown and not Angstrom, cannot proceed reliably
            raise NotImplementedError(
                f"Error 8.1 (check_wavelengthflux): Wavelength unit conversion from '{wl_unit}' to Angstrom not implemented."
            )

    # --- Check 2: Wavelength Range Verification (against metadata) ---
    # Use np.isclose for robust floating-point comparison.
    # Tolerance is scaled by spectral resolution (del_lambda).
    tolerance = SCALE_FACTOR * del_lambda
    if not np.isclose(wl_angstrom[0], lambda_min, atol=tolerance):
        raise ValueError(
            f"Error 9.1 (check_wavelengthflux): First wavelength {wl_angstrom[0]:.2f} Å "
            f"doesn't match expected lambda_min {lambda_min:.2f} Å (within tolerance {tolerance:.2f} Å)."
        )
    if not np.isclose(wl_angstrom[-1], lambda_max, atol=tolerance):
        raise ValueError(
            f"Error 9.2 (check_wavelengthflux): Last wavelength {wl_angstrom[-1]:.2f} Å "
            f"doesn't match expected lambda_max {lambda_max:.2f} Å (within tolerance {tolerance:.2f} Å)."
        )

    # --- Check 3: Wavelength Monotonicity ---
    if not check_wavelength_increasing(wl_angstrom):
        # Check performed *after* potential unit conversion
        raise ValueError(
            "Error 9.3 (check_wavelengthflux): Wavelength values are not strictly monotonically increasing."
        )

    # --- Check 4: Wavelength Physical Plausibility ---
    # Check against hand-defined plausible range for typical optical/NIR spectra.
    if wl_angstrom[0] < HANDCRAFTED_WL_MIN:
        raise ValueError(
            f"Error 10.1 (check_wavelengthflux): First wavelength {wl_angstrom[0]:.2f} Å "
            f"is below minimum plausible value {HANDCRAFTED_WL_MIN} Å."
        )
    if wl_angstrom[-1] > HANDCRAFTED_WL_MAX:
        raise ValueError(
            f"Error 10.2 (check_wavelengthflux): Last wavelength {wl_angstrom[-1]:.2f} Å "
            f"exceeds maximum plausible value {HANDCRAFTED_WL_MAX} Å."
        )

    # --- Check 5: Flux Unit Validation ---
    if spec_unit == "erg cm(-2) sec(-1) Ang(-1)":
        # Units are explicitly the standard ones we expect. Pass.
        # Could add an optional sanity check here using check_flux_units_erg_cm2_s_ang if desired.
        pass
    elif spec_unit == "Other":
        # Units are marked as 'Other' in metadata. Perform heuristic check.
        if not check_flux_units_erg_cm2_s_ang(flux):
            median_abs_flux = np.nanmedian(
                np.abs(flux)
            )  # Recalculate for error message
            raise ValueError(
                f"Error 11.1 (check_wavelengthflux): Flux units are 'Other' but fail plausibility check "
                f"for 'erg cm(-2) s(-1) Å(-1)'. Median absolute flux {median_abs_flux:.2e} is outside the "
                f"expected range ({FLUX_UNIT_CHECK_MIN_MAG:.1e} - {FLUX_UNIT_CHECK_MAX_MAG:.1e}) "
                f"or signal positivity is too low (<{MIN_POSITIVE_FLUX_FRACTION_UNIT_CHECK * 100}%)."
            )
        # else:
        #     # Optional: Inform user that 'Other' units passed the heuristic check
        #     # warnings.warn(
        #     #     "Flux units marked as 'Other' passed the plausibility check for erg cm^-2 s^-1 Å^-1.",
        #     #     UserWarning,
        #     # )
    else:
        # Handle specific known units that require conversion (e.g., Jy, mJy) or fail if unknown.
        # Example placeholder:
        # if spec_unit == "mJy":
        #     # flux_converted = convert_mJy_to_flam(flux, wl_angstrom)
        #     # flux = flux_converted # Replace flux with converted values for subsequent checks
        #     raise NotImplementedError("Flux conversion from mJy not yet implemented.")
        # else:
        raise NotImplementedError(
            f"Error 11.2 (check_wavelengthflux): Spectral unit '{spec_unit}' is not recognized "
            f"or conversion/validation logic is not implemented."
        )

    # --- Check 6: Flux Positivity ---
    # Ensure a sufficient fraction of the flux is positive (avoids pure noise, over-subtraction issues).
    if not check_flux_positive(flux):
        # Calculate actual fraction for a more informative error message
        with warnings.catch_warnings():  # Suppress potential warning from nanmean
            warnings.simplefilter("ignore", category=RuntimeWarning)
            positive_fraction = np.nanmean(flux > 0) if len(flux) > 0 else 0
        raise ValueError(
            f"Error 12.1 (check_wavelengthflux): Flux signal is not sufficiently positive. "
            f"Required >{MIN_POSITIVE_FLUX_FRACTION*100:.0f}% positive fraction, found {positive_fraction*100:.1f}%."
        )

    # --- Success ---
    # All checks passed. (Return value wl_angstrom isn't used by caller currently, but kept for potential future use)
    return wl_angstrom


# ==============================================================================
# ==                      MAIN WRAPPER FUNCTION                               ==
# ==============================================================================


def check_and_plot_spectra(
    spectra_fn,
    meta_df,
    spectra_dir="../1. download ALL wise data/wiserep_data/spectra/",
    plot=True,
):
    """
    Load, validate, and optionally plot a single astronomical spectrum.

    This function orchestrates the process:
    1. Locates the spectrum file.
    2. Looks up corresponding metadata.
    3. Reads the spectrum file using `read_spectra`.
    4. Extracts necessary metadata parameters.
    5. Validates the data using `check_spectrafile`.
    6. If `plot=True` and validation succeeds, generates a plot.

    Handles errors gracefully at each step, printing informative messages.
    """
    spectra_filepath = os.path.join(spectra_dir, spectra_fn)

    try:
        # --- Step 1: Metadata Lookup ---
        # Ensure the DataFrame index is set to 'Ascii file' for efficient lookup.
        # Use a copy if modification is needed to avoid side effects outside the function.
        if meta_df.index.name != "Ascii file":
            meta_df_indexed = meta_df.set_index("Ascii file", drop=False)  # Keep column
        else:
            meta_df_indexed = meta_df

        try:
            spectra_meta = meta_df_indexed.loc[spectra_fn]
        except KeyError:
            # Specific error for missing metadata entry
            raise ValueError(
                f"Error 13.1 (check_and_plot_spectra): Metadata lookup failed. No entry found for '{spectra_fn}' in the provided meta_df."
            )

        # --- Step 2: File Reading ---
        # `read_spectra` handles multiple formats and raises FileNotFoundError or ValueError (Error 2.x) on failure.
        spectra_df = read_spectra(spectra_filepath)

        # --- Step 3: Extract Metadata for Validation ---
        # Access metadata values *after* confirming lookup and file read were successful.
        try:
            wl_unit = spectra_meta["WL Units"]
            spec_unit = spectra_meta["Spec. units"]
            lambda_min = spectra_meta["Lambda-min"]
            lambda_max = spectra_meta["Lambda-max"]
            del_lambda = spectra_meta["Del-Lambda"]
        except KeyError as e:
            # Handle missing *specific columns* within the found metadata entry
            raise ValueError(
                f"Error processing '{spectra_fn}': Metadata entry found, but required column '{e}' is missing."
            )

        # --- Step 4: Data Validation ---
        # `check_spectrafile` performs comprehensive checks and returns column names.
        # Raises ValueError (Errors 3.x-6.x) or propagates errors from `check_wavelengthflux` (Errors 7.x-12.x) or NotImplementedError.
        wl_col, flux_col = check_spectrafile(
            spectra_df, wl_unit, spec_unit, lambda_min, lambda_max, del_lambda
        )

        # --- Step 5: Plotting (Optional) ---
        if plot:
            try:
                # Extract validated data columns for plotting, converting to float just in case
                x = spectra_df[wl_col].astype(float).values
                y = spectra_df[flux_col].astype(float).values
            except (ValueError, TypeError) as e:
                # Catch potential errors during final data extraction/conversion for plotting
                raise ValueError(
                    f"Error 14.1 (check_and_plot_spectra): Failed to convert validated data to numeric for plotting '{spectra_fn}'. Column: '{e}'. Original error: {type(e).__name__}"
                )

            # Define plot labels (assuming validation implies standard units or successful conversion/plausibility check)
            # TODO: Make labels dynamic based on validated/converted units if necessary
            plot_wl_unit_label = "Wavelength ($\AA$)"
            plot_spec_unit_label = (
                r"Flux (erg cm$^{-2}$ s$^{-1} \AA ^{-1}$)"  # Use raw string for LaTeX
            )

            # Generate the plot
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, lw=1)  # Use slightly thinner line
            plt.xlabel(plot_wl_unit_label)
            plt.ylabel(plot_spec_unit_label)
            plt.title(f"Spectrum: {os.path.basename(spectra_fn)}")
            plt.grid(True, linestyle="--", alpha=0.6)  # Slightly adjust grid style
            plt.tight_layout()  # Adjust layout to prevent labels overlapping
            plt.show()

    # --- Exception Handling Block ---
    # Catch specific errors first, then more general ones.
    except FileNotFoundError as e:
        print(f"❌ Error processing '{spectra_fn}': File not found. {e}")
    except KeyError as e:
        # Catches KeyErrors from the *initial* metadata lookup (.loc) if index wasn't set properly,
        # or potentially from the final plotting data extraction if columns vanished (unlikely).
        # Specific missing columns inside metadata are handled above.
        print(f"❌ Error processing '{spectra_fn}': Metadata key error. {e}")
    except ValueError as e:
        # Catches all ValueErrors raised by read_spectra, check_spectrafile, check_wavelengthflux,
        # metadata lookup (Error 13.1), or plotting data conversion (Error 14.1).
        # The error message 'e' should contain the specific error code (X.Y).
        print(f"❌ Validation Error processing '{spectra_fn}': {e}")
    except NotImplementedError as e:
        # Catches errors related to unimplemented unit conversions (Error 8.1, 11.2).
        print(
            f"❌ Processing Error processing '{spectra_fn}': Feature not implemented. {e}"
        )
    except Exception as e:
        # Catch any other unexpected errors (e.g., plotting library issues, unforeseen file issues).
        print(
            f"❌ An unexpected error occurred while processing '{spectra_fn}': {type(e).__name__} - {e}"
        )


# ==============================================================================
# ==                      FUTURE WORK / NOTES                                 ==
# ==============================================================================

# TODO: Add proper unit tests (pytest?) - Crucial for ensuring robustness against diverse and potentially messy spectral file formats.
# TODO: Refine read_spectra error handling - Catch more specific I/O or parsing errors from astropy/pandas for clearer debugging, especially during batch processing.
# TODO: Implement astropy.units - Using Quantity objects would provide robust unit tracking and automated, safe conversions. Highly recommended for long-term maintainability.
# TODO: Add type hints - Once the structure is stable, add type hints (e.g., using typing module) for improved static analysis and developer understanding.
# TODO: Refine check_wavelengthflux NaN handling - Decide if certain NaNs in flux (e.g., at spectral edges) are acceptable and adjust logic accordingly.
# TODO: Robustness of check_spectrafile row count - The current check is basic; consider edge cases or alternative methods if needed.
# TODO: Dynamic Plot Labels - Update plotting section to use actual units derived from metadata or conversion results rather than assuming Angstrom/erg...
# TODO: Configuration Management - Move constants like HANDCRAFTED_WL_MIN/MAX, magnitude limits, etc., to a config file or the fetched settings dictionary for easier modification.
# TODO: Consider adding logging instead of print statements for errors, especially for batch processing.
# LIMIT TO FEW DAYS BEFORE AND MAXIMUM OF SPECTRA - FOR NOW #<-- Placeholder comment from original code
# def #### <-- Placeholder comment from original code
