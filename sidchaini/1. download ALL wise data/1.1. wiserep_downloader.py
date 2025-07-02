#!/usr/bin/env python3
# wiserep_downloader.py - Download spectral data from WISeREP

import requests
import argparse
import concurrent.futures
import datetime
import logging
import re
import shutil
import time
from pathlib import Path
from queue import Queue
import pandas as pd
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry
import zipfile

# --- WISeREP Parameters ---
# Names for the IDs used by WISeREP's web form.
# Source: HTML from https://www.wiserep.org/search/spectra

OBJ_TYPE_MAP = {
    "Afterglow": "23",
    "AGN": "29",
    "Blazar": "32",
    "Computed-Ia": "1003",
    "Computed-IIb": "1014",
    "Computed-IIn": "1021",
    "Computed-IIP": "1011",
    "Computed-PISN": "1020",
    "CV": "27",
    "FBOT": "66",
    "FRB": "130",
    "Galaxy": "30",
    "Gap": "60",
    "Gap I": "61",
    "Gap II": "62",
    "ILRT": "25",
    "Impostor-SN": "99",
    "Kilonova": "70",
    "LBV": "24",
    "Light-Echo": "40",
    "LRN": "65",
    "M dwarf": "210",
    "NA/Unknown": "998",
    "Nova": "26",
    "Other": "0",
    "QSO": "31",
    "SLSN-I": "18",
    "SLSN-II": "19",
    "SLSN-R": "20",
    "SN": "1",
    "SN I": "2",
    "SN I-faint": "15",
    "SN I-rapid": "16",
    "SN Ia": "3",
    "SN Ia-91bg-like": "103",
    "SN Ia-91T-like": "104",
    "SN Ia-Ca-rich": "118",
    "SN Ia-CSM": "106",
    "SN Ia-pec": "100",
    "SN Ia-SC": "102",
    "SN Iax[02cx-like]": "105",
    "SN Ib": "4",
    "SN Ib-Ca-rich": "115",
    "SN Ib-pec": "107",
    "SN Ib/c": "6",
    "SN Ib/c-Ca-rich": "116",
    "SN Ibn": "9",
    "SN Ibn/Icn": "110",
    "SN Ic": "5",
    "SN Ic-BL": "7",
    "SN Ic-Ca-rich": "117",
    "SN Ic-pec": "108",
    "SN Icn": "109",
    "SN Ien": "114",
    "SN II": "10",
    "SN II-pec": "111",
    "SN IIb": "14",
    "SN IIL": "12",
    "SN IIn": "13",
    "SN IIn-pec": "112",
    "SN IIP": "11",
    "Std-spec": "50",
    "TDE": "120",
    "TDE-H": "121",
    "TDE-H-He": "123",
    "TDE-He": "122",
    "Varstar": "28",
    "WD": "190",
    "WR": "200",
    "WR-WC": "202",
    "WR-WN": "201",
    "WR-WO": "203",
}

OBJ_FAMILY_MAP = {
    "CV": "5",
    "FRB": "9",
    "Galaxy": "6",
    "Gap": "7",
    "GRB": "8",
    "GW": "10",
    "High-Energy": "20",
    "Nova": "4",
    "Nuclear": "3",
    "Other": "0",
    "SN": "1",
    "Star": "2",
    "Synthetic": "100",
}

INSTRUMENT_MAP = {
    "AAT / AAOmega-2DF": "236",
    "AAT / RGO": "157",
    "ANU-2.3m / WiFeS": "115",
    "APO-3.5m / APO-TSPEC": "141",
    "APO-3.5m / ARCES": "201",
    "APO-3.5m / DIS": "70",
    "Arecibo / ALFA": "228",
    "Arecibo / L-Wide": "229",
    "ASASSN-1 / Brutus": "153",
    "ASASSN-2 / Cassius": "154",
    "ASASSN-3 / Paczynski": "191",
    "ASASSN-4 / Leavitt": "192",
    "ASASSN-5 / Payne-Gaposchkin": "195",
    "ASKAP / 900MHz": "235",
    "ASKAP / Coherent": "234",
    "ASKAP / FlysEye": "232",
    "ASKAP / Incoherent": "233",
    "AST3 / AST3-Cam": "169",
    "ATLAS-CHL / ATLAS-04": "256",
    "ATLAS-HKO / ATLAS-02": "159",
    "ATLAS-MLO / ATLAS-01": "160",
    "ATLAS-STH / ATLAS-03": "255",
    "BAO-0.6m / BATC": "167",
    "BAO-0.85m / CCD": "77",
    "BAO-2.16m / Cassegrain": "80",
    "BAO-2.16m / Phot-spec": "55",
    "BG2 / BlackGEM-Cam2": "215",
    "BG3 / BlackGEM-Cam3": "216",
    "BG4 / BlackGEM-Cam4": "217",
    "BL41 / FOSC-E5535": "286",
    "Bok / BC-Bok": "68",
    "BTA-6 / SCORPIO": "131",
    "CA-2.2m / CAFOS": "48",
    "CA-3.5m / MOSCA": "60",
    "CA-3.5m / PMAS": "81",
    "CA-3.5m / TWIN": "176",
    "CFHT / MegaCam": "202",
    "CHIME / FRB": "222",
    "CHIME / PULSAR": "223",
    "CMO-2.5m / TDS": "249",
    "CNEOST / STA1600": "161",
    "CrAO / AZT-8": "204",
    "Crossley / PNS": "79",
    "CSS-0.7m / CSS-0.7m-CCD": "276",
    "CTIO-0.41 / CTIO-Photomultiplier": "205",
    "CTIO-0.9 / CASS-DI": "88",
    "CTIO-1.0 / CTIO-Other": "206",
    "CTIO-1.5m / LORAL": "86",
    "CTIO-1.5m / RC-Spec-1.5": "73",
    "CTIO-4m / ARCoIRIS": "173",
    "CTIO-4m / COSMOS": "174",
    "CTIO-4m / DECAM": "172",
    "CTIO-4m / IR-Spec": "87",
    "CTIO-4m / RC-Spec-4": "72",
    "Danish-1.54m / DFOSC": "45",
    "DDO / Cass": "53",
    "DSA-110 / DSA-10": "239",
    "Effelsberg / PSRIX": "240",
    "Ekar / AFOSC": "37",
    "Ekar / BC-Ekar": "38",
    "ElSauce-1m / LISA": "263",
    "ESO-1.5m / BC-ESO": "54",
    "ESO-1m / QUEST": "158",
    "ESO-2.2m / EFOSC-2.2": "33",
    "ESO-2.2m / GROND": "210",
    "ESO-3.6m / EFOSC2-3.6": "32",
    "ESO-NTT / EFOSC2-NTT": "31",
    "ESO-NTT / EMMI": "30",
    "ESO-NTT / Sofi": "34",
    "ESO-NTT / SoXS": "288",
    "Euclid / NISP": "281",
    "Euclid / VIS": "280",
    "FAST / FAST-MB": "238",
    "FLWO-1.5m / FAST": "44",
    "FLWO-1.5m / TRES": "144",
    "FTN / EM01": "110",
    "FTN / FLOYDS-N": "108",
    "FTN / FS02": "112",
    "FTS / FLOYDS-S": "125",
    "FTS / FS01": "105",
    "Gaia / Gaia-astrometric": "162",
    "Gaia / Gaia-photometric": "163",
    "Gaia / Gaia-RVS": "164",
    "GALEX / GALEX": "102",
    "Galileo / AVI": "93",
    "Galileo / BC-Asi": "57",
    "GAO-0.65m / GCS": "133",
    "GAO-1.5m / GLOWS": "65",
    "Gattini / Gattini-camera": "220",
    "GBT / GUPPI": "227",
    "Gemini-N / GMOS": "6",
    "Gemini-N / GNIRS": "166",
    "Gemini-S / Flamingos-2": "197",
    "Gemini-S / GMOS-S": "9",
    "GMRT / GMRT-Band3": "242",
    "GOTO-N / GOTO-1": "218",
    "GOTO-N / GOTO-2": "264",
    "GOTO-S / GOTO-3": "265",
    "GOTO-S / GOTO-4": "266",
    "GTC / EMIR": "270",
    "GTC / HiPERCAM": "271",
    "GTC / MEGARA": "272",
    "GTC / OSIRIS": "101",
    "Harlan-Smith / ES2": "89",
    "Harlan-Smith / IDS-McDonald": "171",
    "Harlan-Smith / LCS": "91",
    "HCT-2m / HFOSC": "46",
    "HET / HET-HRS": "123",
    "HET / HET-LRS": "43",
    "HET / HET-MRS": "124",
    "HST / STIS": "83",
    "HST / WFC3": "194",
    "IGO / IFOSC": "184",
    "ILMT / 4K-IMG3": "274",
    "INT-2.5m / FOS": "67",
    "INT-2.5m / IDS": "19",
    "IRiS / IRiS": "262",
    "IRTF / SpeX": "122",
    "IUE / IUE": "94",
    "JAST80 / T80Cam": "284",
    "JST250 / JPCam": "283",
    "JWST / MIRI": "188",
    "JWST / NIRCam": "185",
    "JWST / NIRISS": "187",
    "JWST / NIRSpec": "186",
    "KAIT / KAITCam": "203",
    "Kanata / HOWPol": "138",
    "KAO / LOSA-F2": "142",
    "Keck1 / HIRES": "82",
    "Keck1 / LRIS": "3",
    "Keck1 / MOSFIRE": "130",
    "Keck2 / DEIMOS": "4",
    "Keck2 / ESI": "100",
    "Keck2 / KCWI": "259",
    "Keck2 / NIRES": "252",
    "KPNO-2.1m / IDS-KPNO": "212",
    "LAMOST / LRSs": "251",
    "LAST / LAST-Cam": "269",
    "LBT / LUCI": "128",
    "LBT / MODS1": "120",
    "LBT / MODS2": "121",
    "LCO-duPont / BC-duPont": "63",
    "LCO-duPont / Mod-spec": "64",
    "LCO-duPont / WFCCD": "62",
    "LCO1m / Sinistro": "208",
    "LCO2m / Spectral": "209",
    "LDT / Deveny-LMI": "143",
    "Lesedi / Mookodi": "279",
    "Lick-3m / KAST": "10",
    "Lick-3m / ShARCS": "198",
    "Lick-3m / UV-Schmidt": "99",
    "Lick-3m / VNIRIS": "199",
    "Lick1m / Nickel-Spec": "39",
    "Lijiang-2.4m / YFOSC": "107",
    "LPA / LPA": "230",
    "LT / FRODOspec": "95",
    "LT / IO-I": "245",
    "LT / IO-O": "244",
    "LT / SPRAT": "156",
    "Magellan-Baade / BC-Magellan": "151",
    "Magellan-Baade / FIRE": "116",
    "Magellan-Baade / IMACS": "75",
    "Magellan-Baade / MagE": "137",
    "Magellan-Clay / LDSS-2": "69",
    "Magellan-Clay / LDSS-3": "78",
    "Magellan-Clay / MIKE": "84",
    "Mayall / DESI": "258",
    "Mayall / KOSMOS": "200",
    "Mayall / RC-Spec": "5",
    "MDM-2.4 / BC-OSU": "59",
    "MDM-2.4 / MARK-III": "85",
    "MDM-2.4 / modspec": "135",
    "MDM-2.4 / OSMOS": "150",
    "MDM-2.4 / TIFKAM": "170",
    "Mephisto / Mephisto-Cam": "278",
    "Mercator / HERMES": "145",
    "ML1 / MeerLICHT-Cam": "214",
    "MLO-1.5m / ITS-MLO": "213",
    "MLO-1.5m / SN110-106": "275",
    "MLO-1m / CCD-MLO": "113",
    "MMT / BINOSPEC": "221",
    "MMT / Hectospec": "96",
    "MMT / MMIRS": "180",
    "MMT / MMT-Blue": "58",
    "MMT / MMT-Red": "98",
    "MOST / UTMOST": "231",
    "MSO-74in / BC-MSO": "132",
    "Mt-Abu / Abu-NICS": "165",
    "Nayuta / MALLS": "49",
    "NOT / ALFOSC": "41",
    "NOT / FIES": "267",
    "NOT / NOTCam": "254",
    "NOT / StanCam": "207",
    "OHP-1.93m / Carelec": "106",
    "OHP-1.93m / MISTRAL": "261",
    "Other / Other": "0",
    "Other / Synthetic": "268",
    "Otto-Struve / IGS": "92",
    "P200 / DBSP": "1",
    "P200 / LFC": "2",
    "P200 / NGPS": "285",
    "P200 / P200-TSPEC": "109",
    "P48 / CFH12k": "103",
    "P48 / ZTF-Cam": "196",
    "P60 / P60-Cam": "104",
    "P60 / SEDM": "149",
    "Parkes / MB20": "225",
    "Parkes / UWL": "243",
    "Plaskett / Plaskett": "146",
    "PO-RC32 / FLI-CCD": "273",
    "PS1 / GPC1": "155",
    "PS2 / GPC2": "257",
    "RAO-BN / RAO-BN-Cam": "253",
    "Rubin / LSSTCam": "287",
    "SAAO / G-Spec": "119",
    "SALT / HRS-SALT": "118",
    "SALT / RSS": "117",
    "SkyMapper / SM-WFCam": "152",
    "Sloan / BOSS": "250",
    "Sloan / SDSS-Spec": "140",
    "SOAR / Goodman": "127",
    "SOAR / SOAR-OSIRIS": "136",
    "SOAR / TripleSpec": "260",
    "SPM15 / RATIR": "126",
    "SRT / LP-DualBand": "241",
    "SSO-2.3m / DBS": "61",
    "SST / IRAC": "193",
    "SST / IRS": "66",
    "Subaru / FOCAS": "71",
    "Subaru / HDS": "175",
    "Subaru / HSC": "177",
    "Subaru / IRCS": "178",
    "Super-Kamiokande / Super-Kamiokande": "248",
    "Swift-UVOT / UV-grism": "50",
    "Swift-UVOT / UVOT-Imager": "52",
    "Swift-UVOT / V-grism": "51",
    "TMT / CCD-TMT": "219",
    "TNG / DOLORES": "15",
    "TNG / HARPS-N": "147",
    "TNG / NICS": "40",
    "TNG / SARG": "134",
    "UH88 / SNIFS": "11",
    "UKIRT / CGS4": "47",
    "UTMOST-EW / MPSR": "246",
    "UTMOST-NS / NPSR": "247",
    "VLA / WIDAR": "226",
    "VLT-UT1 / FORS1": "8",
    "VLT-UT1 / FORS2": "7",
    "VLT-UT2 / FORS1-UT2": "76",
    "VLT-UT2 / UVES": "148",
    "VLT-UT2 / X-Shooter": "12",
    "VLT-UT3 / ISAAC": "35",
    "VLT-UT3 / VIMOS": "168",
    "VLT-UT4 / MUSE": "182",
    "VLT-UT4 / SINFONI": "183",
    "WFST / WFST-WFC": "282",
    "WHT-4.2m / ACAM": "97",
    "WHT-4.2m / FOS-1": "18",
    "WHT-4.2m / FOS-2": "17",
    "WHT-4.2m / ISIS": "16",
    "WHT-4.2m / LIRIS": "129",
    "Wise-C18 / C18-Cam": "114",
    "Wise-C28 / C28-Cam": "277",
    "Wise1m / FOSC": "22",
    "Wise1m / Laiwo": "20",
    "Wise1m / PI": "21",
    "WIYN / Hydra": "74",
    "WSRT / Apertif": "237",
    "XLT / BFOSC": "56",
    "XLT / HRS-XLT": "181",
    "XLT / OMR": "179",
}

SPEC_TYPE_MAP = {
    "Object": "10",
    "Host": "20",
    "Sky": "30",
    "Arcs": "40",
    "Synthetic": "50",
}

QUALITY_MAP = {"Low": "1", "Medium": "2", "High": "3"}

GROUP_MAP = {
    "None": "0",
    "ASAS-SN": "38",
    "Asiago": "24",
    "AZTEC": "81",
    "BlackGem": "85",
    "BSNIP": "16",
    "BTDG": "63",
    "CCCP": "2",
    "CET-3PO": "92",
    "CfA-Ia": "18",
    "CfA-Stripped": "45",
    "CSP": "26",
    "DESI": "78",
    "ENGRAVE": "59",
    "ePESSTO": "23",
    "ePESSTO+": "60",
    "ESO-NTT": "20",
    "ESO-SoXS": "96",
    "Fink follow-up": "82",
    "Giorgos-Space": "14",
    "Golden SE-SNe ": "75",
    "GOTO": "90",
    "Groh_Boian": "61",
    "GSP": "58",
    "HIRES": "5",
    "HST-17205-cycle30": "77",
    "HST-Cy19-Foley": "30",
    "HST-Ia": "6",
    "IAC-Time-Domain": "89",
    "iPTF": "15",
    "Jerkstrand": "51",
    "Kinder": "65",
    "KISS": "47",
    "KITS": "80",
    "LOSS": "50",
    "MOST Hosts": "83",
    "MUSSES": "72",
    "NOIRlab NordicOpticalTelescope": "84",
    "NOT-ZTF": "62",
    "NTT-NOT": "19",
    "NUTS2": "79",
    "Onori's spectra": "94",
    "PESSTO": "22",
    "PESSTO_SSDR1-4": "76",
    "PESSTO-SSDR1-2": "32",
    "POISE": "67",
    "PS1": "11",
    "PTF": "1",
    "PTF-Ia": "4",
    "PTFdryrun": "3",
    "Radio-Horesh": "13",
    "SCAT": "68",
    "SDSS TOO": "95",
    "SDSS-SNe": "48",
    "Sladjana-Novae": "49",
    "SN spectra": "69",
    "SN-latetime": "74",
    "SN-Polarimetry": "57",
    "SNfactory": "42",
    "SNLS": "40",
    "StarDestroyers": "87",
    "STARGATE": "54",
    "SUSPECT": "10",
    "TCD": "91",
    "TNS": "53",
    "Tomo-e": "73",
    "TS3": "44",
    "UCB-SNDB": "17",
    "ULTRASAT-LT": "93",
    "UoB Transients": "64",
    "WIS_Astro": "55",
    "WIS-Flashers": "66",
    "WOOTS": "9",
    "WTFU": "25",
    "YSE": "71",
    "ZEHUI PENG": "86",
    "ZTF": "52",
    "ZTF SN Ia DR2": "88",
    "ZTF-PESSTO-Joint": "56",
    "ZTF-SLSN": "70",
}


# --- Core Functions ---


def setup_logging(datadir: Path) -> logging.Logger:
    """Sets up logging to file and console."""
    log_file = (
        datadir
        / f"download_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def get_session() -> requests.Session:
    """Creates a requests session with a retry strategy."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        }
    )
    return session


def find_downloaded_pages(datadir: Path) -> tuple[list[int], list[int], int]:
    """Finds all downloaded pages, identifies gaps, and returns the max page."""
    all_files = [
        f
        for f in datadir.iterdir()
        if f.name.startswith("wiserep_spectra_page") and f.name.endswith(".csv")
    ]

    if not all_files:
        return [], [], 0

    try:
        page_numbers = sorted([int(f.stem.split("page")[1]) for f in all_files])
        if not page_numbers:
            return [], [], 0

        max_page = max(page_numbers)
        expected_pages = set(range(1, max_page + 1))
        missing_pages = sorted(list(expected_pages - set(page_numbers)))

        return page_numbers, missing_pages, max_page
    except (ValueError, IndexError):
        return [], [], 0


def download_wiserep_data(
    page: int,
    datadir: Path,
    session: requests.Session,
    logger: logging.Logger,
    args: argparse.Namespace,
) -> bool:
    """
    Downloads, extracts, and processes data for a single page from WISeREP.
    """
    base_url = "https://www.wiserep.org/search/spectra"
    params = [
        ("format", "csv"),
        ("files_type", "ascii"),
        ("num_page", "50"),
        ("page", str(page)),
    ]

    # --- Apply filters from command-line arguments ---
    if args.public:
        params.append(("public", args.public))
    if args.added_within_value and args.added_within_units:
        params.append(("inserted_period_value", str(args.added_within_value)))
        params.append(("inserted_period_units", args.added_within_units))
    if args.obj_type:
        params.extend(("type[]", OBJ_TYPE_MAP[t]) for t in args.obj_type)
    if args.obj_family:
        params.extend(("type_family[]", OBJ_FAMILY_MAP[f]) for f in args.obj_family)
    if args.redshift:
        params.append(("redshift_min", str(args.redshift[0])))
        params.append(("redshift_max", str(args.redshift[1])))
    if args.instruments:
        params.extend(("instruments[]", INSTRUMENT_MAP[i]) for i in args.instruments)
    if args.spec_types:
        params.extend(("spectype[]", SPEC_TYPE_MAP[s]) for s in args.spec_types)
    if args.quality:
        params.extend(("qualityid[]", QUALITY_MAP[q]) for q in args.quality)
    if args.source_groups:
        params.extend(("groupid[]", GROUP_MAP[g]) for g in args.source_groups)
    if args.creator:
        params.append(("creation_modifier", args.creator))

    try:
        response = session.get(base_url, params=params, stream=True)
        response.raise_for_status()
        logger.info(f"Downloading from Page {page}: {response.url}")

        content_type = response.headers.get("Content-Type", "").lower()
        is_zip = (
            "application/zip" in content_type
            or "application/x-zip-compressed" in content_type
        )

        if not is_zip:
            # Fallback check for zip file signature
            content_peek = response.raw.peek(4)
            if content_peek == b"PK\x03\x04":
                is_zip = True
            else:
                logger.warning(
                    f"Response for page {page} is not a zip file. Content-Type: {content_type}. Skipping."
                )
                return False

        page_dir = datadir / f"page_{page}"
        page_dir.mkdir(exist_ok=True)
        zip_filename = page_dir / f"wiserep_page{page}.zip"

        with open(zip_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(page_dir)

        csv_source = page_dir / "wiserep_spectra.csv"
        csv_dest = datadir / f"wiserep_spectra_page{page}.csv"

        if csv_source.exists():
            shutil.move(csv_source, csv_dest)
            logger.info(f"Successfully downloaded metadata for page {page}")

            spectra_dir = datadir / "spectra"
            spectra_dir.mkdir(exist_ok=True)

            spectrum_files = list(page_dir.iterdir())
            for spec_file in spectrum_files:
                if spec_file.name.endswith(".zip"):
                    continue
                dest = spectra_dir / spec_file.name
                if dest.exists():
                    dest = (
                        spectra_dir / f"{spec_file.stem}_page{page}{spec_file.suffix}"
                    )
                shutil.move(spec_file, dest)

            if spectrum_files:
                logger.info(
                    f"Moved {len(spectrum_files)} spectrum files from page {page}"
                )
        else:
            logger.warning(
                f"CSV file not found after extraction for page {page}. Likely no data on this page."
            )
            shutil.rmtree(page_dir)
            zip_filename.unlink(missing_ok=True)
            return False

        shutil.rmtree(page_dir)
        zip_filename.unlink(missing_ok=True)

        try:
            df = pd.read_csv(csv_dest)
            if df.empty:
                logger.warning(
                    f"Page {page} returned an empty dataset. This might be the last page."
                )
                return False
            return True
        except pd.errors.EmptyDataError:
            logger.warning(f"Empty CSV file for page {page}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading data for page {page}: {e}")
        return False
    except zipfile.BadZipFile:
        logger.warning(
            f"Invalid or empty zip file for page {page}. Possibly no spectra available."
        )
        if "zip_filename" in locals() and zip_filename.exists():
            zip_filename.unlink()
        if "page_dir" in locals() and page_dir.exists():
            shutil.rmtree(page_dir)
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing page {page}: {e}", exc_info=True)
        return False


def download_all_wiserep_data(
    datadir: Path, logger: logging.Logger, args: argparse.Namespace
):
    """
    Manages concurrent downloading of data from all specified pages.
    """
    session = get_session()
    logger.info(
        f"Starting WISeREP data download with {args.threads} threads. Filters: {vars(args)}"
    )

    start_page = args.start
    end_page = float("inf") if args.max is None else start_page + args.max - 1

    results = {}
    consecutive_empty_pages = 0
    MAX_CONSECUTIVE_EMPTY = 5

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        current_page = start_page
        futures = {}

        with tqdm(
            desc="Downloading pages", unit="page", total=args.max if args.max else None
        ) as pbar:
            while current_page <= end_page:
                # Fill the executor with new tasks up to the thread limit
                while len(futures) < args.threads and current_page <= end_page:
                    future = executor.submit(
                        download_wiserep_data,
                        current_page,
                        datadir,
                        session,
                        logger,
                        args,
                    )
                    futures[future] = current_page
                    current_page += 1
                    # A small delay to avoid overwhelming the server with initial requests
                    time.sleep(args.delay / args.threads)

                # Wait for at least one future to complete
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    page = futures.pop(future)
                    try:
                        success = future.result()
                        results[page] = success
                        pbar.update(1)

                        if not success:
                            # Check for consecutive failures
                            last_pages = sorted(results.keys())[-MAX_CONSECUTIVE_EMPTY:]
                            if len(last_pages) == MAX_CONSECUTIVE_EMPTY and all(
                                not results.get(p) for p in last_pages
                            ):
                                logger.info(
                                    f"Detected {MAX_CONSECUTIVE_EMPTY} consecutive empty pages. Stopping."
                                )
                                # Cancel remaining futures
                                for f in futures:
                                    f.cancel()
                                end_page = 0  # Break outer loop
                                break

                    except Exception as e:
                        logger.error(f"Thread error processing page {page}: {e}")
                        results[page] = False

    successful_pages = sum(1 for success in results.values() if success)
    logger.info(f"Download complete. Successfully processed {successful_pages} pages.")


def verify_and_combine_data(
    datadir: Path, logger: logging.Logger
) -> pd.DataFrame | None:
    """Verifies downloaded CSV files and combines them into a single dataset."""
    all_files = sorted(
        [f for f in datadir.glob("wiserep_spectra_page*.csv")],
        key=lambda f: int(f.stem.split("page")[-1]),
    )

    if not all_files:
        logger.warning("No CSV files found to combine.")
        return None

    logger.info(f"Found {len(all_files)} CSV files to combine.")

    df_list = []
    for file in tqdm(all_files, desc="Combining files"):
        try:
            df = pd.read_csv(file)
            if df.empty:
                logger.warning(f"File {file.name} is empty and will be skipped.")
                continue
            df_list.append(df)
        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")

    if not df_list:
        logger.warning("No valid data found in CSV files to combine.")
        return None

    combined_df = pd.concat(df_list, ignore_index=True)

    if "Unnamed: 0" in combined_df.columns:
        combined_df = combined_df.rename(columns={"Unnamed: 0": "Obj. ID"})

    logger.info(f"Initial combined dataset has {len(combined_df)} rows.")

    if combined_df.empty:
        logger.warning("Combined dataframe is empty.")
        return None

    # before_dedup = len(combined_df)
    # dedup_subset = ["Obj. ID", "Ascii file"]
    # if all(col in combined_df.columns for col in dedup_subset):
    #     combined_df.drop_duplicates(subset=dedup_subset, inplace=True)
    #     after_dedup = len(combined_df)
    #     if before_dedup > after_dedup:
    #         logger.info(f"Removed {before_dedup - after_dedup} duplicate entries.")
    # else:
    #     missing_cols = [col for col in dedup_subset if col not in combined_df.columns]
    #     logger.warning(
    #         f"Skipping deduplication because columns {missing_cols} were not found. "
    #         f"Available columns: {list(combined_df.columns)}"
    #     )

    output_file = Path("wiserep_spectra_combined.csv")

    # Clean up and format columns
    if "Obj. ID" in combined_df.columns:
        combined_df = combined_df.rename(columns={"Obj. ID": "wise_objid"})
        if "JD" in combined_df.columns:
            combined_df = combined_df.sort_values(by=["wise_objid", "JD"])
        else:
            logger.warning("Column 'JD' not found. Sorting by 'wise_objid' only.")
            combined_df = combined_df.sort_values(by=["wise_objid"])

    for col in ["IAU name", "Publish", "Remarks", "Created by"]:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str).str.strip()

    combined_df.to_csv(output_file, index=False)
    logger.info(
        f"Combined {len(df_list)} files with {len(combined_df)} total rows into {output_file}"
    )

    return combined_df


def setup_arg_parser() -> argparse.ArgumentParser:
    """Configures the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download spectral data from WISeREP (https://www.wiserep.org).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
1. Download all spectra for SN Ia:
   python wiserep_downloader.py --obj_type "SN Ia"

2. Download spectra from specific instruments, starting from page 10:
   python wiserep_downloader.py --instruments "Keck1 / LRIS" "P60 / SEDM" --start 10

3. Download high-quality spectra from the ZTF source group:
   python wiserep_downloader.py --quality High --source_groups ZTF
""",
    )

    # --- Execution Control ---
    control_group = parser.add_argument_group("Execution Control")
    control_group.add_argument(
        "-d",
        "--datadir",
        type=Path,
        default=Path("wiserep_data"),
        help="Directory to store downloaded data (default: wiserep_data).",
    )
    control_group.add_argument(
        "-s",
        "--start",
        type=int,
        default=None,
        help="Starting page number (default: resume from the last downloaded page).",
    )
    control_group.add_argument(
        "-m",
        "--max",
        type=int,
        default=None,
        help="Maximum number of pages to download.",
    )
    control_group.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between new download requests in seconds (default: 0.2).",
    )
    control_group.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Number of concurrent download threads (default: 10).",
    )
    control_group.add_argument(
        "--combine-only",
        action="store_true",
        help="Only combine existing CSV files, do not download new data.",
    )
    control_group.add_argument(
        "--organize-only", action="store_true", help="Alias for --combine-only."
    )

    # --- WISeREP Search Filters ---
    filter_group = parser.add_argument_group("WISeREP Search Filters")
    filter_group.add_argument(
        "--public",
        choices=["yes", "no", "all"],
        default="all",
        help="Filter by public status.",
    )
    filter_group.add_argument(
        "--added_within_value",
        type=int,
        help="Filter spectra added in the last X units (e.g., 30).",
    )
    filter_group.add_argument(
        "--added_within_units",
        choices=["days", "months", "years"],
        help="Units for --added_within_value.",
    )
    filter_group.add_argument(
        "--obj_type",
        nargs="+",
        choices=OBJ_TYPE_MAP.keys(),
        help="Filter by object type(s). Can provide multiple. Example: --obj_type 'SN Ia' 'SN IIb'",
    )
    filter_group.add_argument(
        "--obj_family",
        nargs="+",
        choices=OBJ_FAMILY_MAP.keys(),
        help="Filter by object family (e.g., 'SN', 'TDE').",
    )
    filter_group.add_argument(
        "-z",
        "--redshift",
        nargs=2,
        type=float,
        help="Filter by redshift range [min max].",
    )
    filter_group.add_argument(
        "-i",
        "--instruments",
        nargs="+",
        choices=INSTRUMENT_MAP.keys(),
        help="Filter by instrument(s).",
    )
    filter_group.add_argument(
        "--spec_types",
        nargs="+",
        choices=SPEC_TYPE_MAP.keys(),
        help="Filter by spectrum type(s).",
    )
    filter_group.add_argument(
        "--quality",
        nargs="+",
        choices=QUALITY_MAP.keys(),
        help="Filter by spectrum quality.",
    )
    filter_group.add_argument(
        "--source_groups",
        nargs="+",
        choices=GROUP_MAP.keys(),
        help="Filter by source group(s).",
    )
    filter_group.add_argument(
        "-c", "--creator", help="Filter by creator username (e.g., 'TNS_Bot1')."
    )

    return parser


def main():
    """Main execution function."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    args.datadir.mkdir(exist_ok=True)
    logger = setup_logging(args.datadir)

    if args.organize_only:
        args.combine_only = True

    if not args.combine_only:
        _, missing_pages, max_page = find_downloaded_pages(args.datadir)

        # First, download any pages that are missing from the sequence.
        if missing_pages:
            logger.info(f"Found {len(missing_pages)} missing pages: {missing_pages}")
            logger.info("Downloading missing pages first...")
            session = get_session()
            for page in tqdm(missing_pages, desc="Downloading missing pages"):
                download_wiserep_data(page, args.datadir, session, logger, args)
                time.sleep(args.delay)

        # Determine the starting page for new downloads.
        if args.start is None:
            args.start = max_page
            logger.info(f"Resuming download from page {args.start}")

        # Download new pages.
        download_all_wiserep_data(args.datadir, logger, args)

    # Finally, combine all downloaded data.
    verify_and_combine_data(args.datadir, logger)

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
