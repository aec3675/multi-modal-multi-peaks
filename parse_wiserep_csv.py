# Snippet to parse and merge WISeRep output CSV files
import glob
from astropy.io import ascii
from astropy.table import vstack, Table
import numpy as np

filenames = glob.glob("wis*csv")

cols = "'Obj. ID', 'IAU name', 'Internal name/s', 'Obj. RA', 'Obj. DEC', 'Obj. Type', 'Redshift', 'Spec. ID', 'Obs-date', 'JD', 'Phase (days)', 'From', 'Telescope', 'Instrument', 'Observer/s', 'Reducer/s', 'Source group', 'Public', 'Associated groups', 'End prop. period', 'Ascii file', 'Fits file', 'Spec. type', 'Spec. quality', 'Extinction-Corrected', 'WL Medium', 'WL Units', 'Flux Unit Coefficient', 'Spec. units', 'Flux Calibrated By', 'Exp-time', 'Aperture (slit)', 'HA', 'Airmass', 'Dichroic', 'Grism', 'Grating', 'Blaze', 'Lambda-min', 'Lambda-max', 'Del-Lambda', 'Contrib', 'Publish', 'Remarks', 'Created by', 'Creation date'".replace("'", "")

with open("all_wiserep_spectra.csv", 'w') as f:
    f.write(f"{cols}\n")

for filename in filenames:
    t = ascii.read(filename, format='csv')
    if len(t) == 0:
        continue
    for i in np.arange(len(t)):
        row = ", ".join( [str(t[cn][i]).replace(","," ") for cn in t.colnames] )
        with open("all_wiserep_spectra.csv", "a") as f:
            f.write(f"{row}\n")

