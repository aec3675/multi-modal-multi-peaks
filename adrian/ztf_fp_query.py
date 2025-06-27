#!/usr/bin/env python3

import requests
import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json
import re
# import wget
import shlex

def submit_post(ra_list, dec_list, jd_start, jd_end):
    ra = json.dumps(ra_list)
    print('ra', ra)
    dec = json.dumps(dec_list)
    print('dec', dec)
    jdstart = json.dumps(jd_start)
    print('JD start', jdstart)
    jdend = json.dumps(jd_end)
    print('JD end', jdend)

    # start JD for all input target positions.
    # end JD for all input target positions.
    email = 'pnr5sh@virginia.edu'       # email you subscribed with.
    userpass = 'vhnp411'                # password that was issued to you.
    payload = {'ra': ra, 'dec': dec,
            'startJD': jdstart, 'endJD': jdend,
            'email': email, 'userpass': userpass}
    # fixed IP address/URL where requests are submitted:
    url = 'https://ztfweb.ipac.caltech.edu/cgi-bin/batchfp.py/submit'
    r = requests.post(url,auth=('ztffps', 'dontgocrazy!'), data=payload)
    print("Status_code=",r.status_code)
    

def check_status(datadir='', download=False, num_objs=float):
    settings = {'email': 'pnr5sh@virginia.edu','userpass': 'vhnp411',
            'option': 'All recent jobs', 'action': 'Query Database'}
    r = requests.get('https://ztfweb.ipac.caltech.edu/cgi-bin/getBatchForcedPhotometryRequests.cgi',
                    auth=('ztffps', 'dontgocrazy!'),params=settings)
    #print(r.text)
    if r.status_code == 200:
        print("Script executed normally and queried the ZTF Batch Forced Photometry database.\n")
        wget_prefix = 'wget --http-user=ztffps --http-passwd=dontgocrazy! -O '
        wget_url = 'https://ztfweb.ipac.caltech.edu'
        wget_suffix = "'"
        lightcurves = re.findall(r'/ztf/ops.+?lc.txt\b',r.text)
        if download:
            if lightcurves is not None:
                for lc in lightcurves[-num_objs:]:
                    url = wget_url + lc
                    savefile = str(datadir)+lc[-29:]
                    # print(savefile)
                    os.system(f"{wget_prefix} {savefile} {shlex.quote(url)}")
            else:
                print("error: lightcurve object is None")
    else:
        print("Status_code=",r.status_code,"; Jobs either queued or abnormal execution.")


def setup_arg_parser() -> argparse.ArgumentParser:
    """Configures the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download forced photometry data from ZTF.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Execution Control ---
    control_group = parser.add_argument_group("Execution Control")
    control_group.add_argument(
        "-d",
        "--datadir",
        type=Path,
        default=Path("ztf_fp_data/"),
        help="Directory to store downloaded data (default: ztf_fp_data).",
    )
    control_group.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="csv file containing ra, dec, jd_spec for each object, new objs separated by line",
    )
    control_group.add_argument(
        "-s",
        "--submitQuery",
        type=str,
        default=False,
        help="submit new query to ZTF FP database",
    )
    control_group.add_argument(
        "-c",
        "--checkStatus",
        type=str,
        default=False,
        help="only check the status of the last batch query, will not re-query",
    )
    control_group.add_argument(
        "-w",
        "--download",
        type=str,
        default=False,
        help="download data from ZTF FP",
    )
    return parser

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    args.datadir.mkdir(exist_ok=True)

    data = pd.read_csv(args.filename, header='infer')

    #converting series to arrays to lists for json
    ras = data['ra'].to_numpy().tolist()
    decs = data['dec'].to_numpy().tolist()
    jd_start = data['jd_start'].to_numpy().tolist()
    jd_end = data['jd_end'].to_numpy().tolist()

    if args.submitQuery=='True':
        print('Submitting query to ZTF...\n')
        print(f"Number of (ra,dec) pairs = {len(ras)}\n")

        if len(ras) != len(decs) != len(jd_start) != len(jd_end):
            print('unequal len of inputs, exiting...')
            exit()
            
        submit_post(ras, decs, jd_start, jd_end)                     #submit query
    
    if args.checkStatus=='True':
        print('Checking status of latest query...\n')
        check_status(datadir=args.datadir,download=False,num_objs=len(ras))            #check query
    
    if args.download=='True':
        print('Downloading data from latest query...\n')
        check_status(datadir=args.datadir,download=args.download,num_objs=len(ras))    #check query + download data

    if not args.submitQuery and not args.checkStatus and not args.download:
        print('No action specified, exiting...')
        exit()
    elif args.submitQuery=='False' and args.checkStatus=='False' and args.download=='False':
        print('No action specified, exiting...')
        exit()

if __name__ == "__main__":
    main()