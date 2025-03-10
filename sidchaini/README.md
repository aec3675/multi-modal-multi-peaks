# Supernova Spectroscopy Analysis Pipeline

This directory contains my reimplementation of the WISeREP download for comprehensive analysis of supernova spectra.

## Directory Structure

### 0. example of spectra - sdss/
Contains example SDSS spectra and notebooks demonstrating how to work with FITS files using AstroPy. This is a reference for myself to see the SDSS spectral format.

### 1. download ALL wise data/
Scripts and notebooks for systematically downloading supernova spectra from the WISeREP repository. 
- `wiserep_downloader.py`: Downloads all spectra.

### 2. read and make single dataset/
This directory will contain scripts for combining and standardizing the SDSS and WISeREP data into a single big dataset (`tensors`).

— Sid Chaini