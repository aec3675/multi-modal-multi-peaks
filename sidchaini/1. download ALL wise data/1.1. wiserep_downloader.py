#!/usr/bin/env python3
# wiserep_downloader.py - Download spectral data from WISeREP

import requests
import zipfile
import os
import shutil
import time
import logging
import datetime
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib.parse
import re
import threading
from queue import Queue
import concurrent.futures


# Set up logging
def setup_logging(datadir):
    log_file = os.path.join(
        datadir, f"download_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def get_session():
    """Create a requests session with retry strategy"""
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


def find_downloaded_pages(datadir):
    """Find all downloaded pages and identify any gaps in the sequence"""
    all_files = [
        f
        for f in os.listdir(datadir)
        if f.startswith("wiserep_spectra_page") and f.endswith(".csv")
    ]

    if not all_files:
        return [], [], 0

    # Extract page numbers
    try:
        page_numbers = [int(f.split("page")[1].split(".")[0]) for f in all_files]
        page_numbers.sort()  # Sort the page numbers

        if not page_numbers:
            return [], [], 0

        # Find gaps in the sequence
        max_page = max(page_numbers)
        expected_pages = set(range(1, max_page + 1))
        downloaded_pages = set(page_numbers)
        missing_pages = list(expected_pages - downloaded_pages)
        missing_pages.sort()  # Sort missing pages

        return page_numbers, missing_pages, max_page
    except (ValueError, IndexError):
        # Handle malformed filenames
        return [], [], 0


def find_last_downloaded_page(datadir):
    """Find the last successfully downloaded page to enable resuming"""
    # For backward compatibility
    _, _, max_page = find_downloaded_pages(datadir)
    return max_page


def download_wiserep_data(
    page,
    datadir,
    session,
    logger,
    obj_type=None,
    redshift_range=None,
    creator=None,
    inst=None,
    tel=None,
):
    """Download WISeREP data with filtering options"""

    # Create query parameters
    params = {
        "format": "csv",
        "files_type": "ascii",
        "num_page": 100,
        "page": int(page),
    }

    # Add filters if specified
    if obj_type:
        params["objtype"] = obj_type  # e.g., "SN Ia"
    if redshift_range:
        params["redshift_min"] = redshift_range[0]
        params["redshift_max"] = redshift_range[1]
    if creator:
        params["creation_modifier"] = creator  # e.g., "TNS_Bot1"
    if inst:
        params["instrument"] = inst
    if tel:
        params["telescope"] = tel

    # Build URL with proper encoding
    base_url = "https://www.wiserep.org/search/spectra?"
    url = base_url + urllib.parse.urlencode(params)

    logger.info(f"Downloading from Page {page}: {url}")

    # Create page-specific directory
    page_dir = os.path.join(datadir, f"page_{page}")
    os.makedirs(page_dir, exist_ok=True)

    zip_filename = os.path.join(page_dir, f"wiserep_page{page}.zip")

    try:
        # Download the file using requests
        response = session.get(url, stream=True)
        response.raise_for_status()

        # Check if response is a zip file
        content_type = response.headers.get("Content-Type", "").lower()
        content_disposition = response.headers.get("Content-Disposition", "").lower()

        # Multiple checks for zip content
        is_zip = (
            "application/zip" in content_type
            or "application/x-zip" in content_type
            or ".zip" in content_disposition
        )

        # Check first bytes if headers don't indicate zip
        if not is_zip and len(response.content) >= 4:
            is_zip = response.content[:4] == b"PK\x03\x04"

        if not is_zip:
            logger.warning(
                f"Response for page {page} is not a zip file. Content-Type: {content_type}"
            )
            return False

        # Save the zip file
        with open(zip_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the zip file to the page directory
        try:
            with zipfile.ZipFile(zip_filename, "r") as zip_ref:
                zip_ref.extractall(page_dir)
        except zipfile.BadZipFile:
            logger.warning(f"Invalid zip file for page {page}")
            if os.path.exists(zip_filename):
                os.remove(zip_filename)
            return False

        # Move and rename the extracted CSV file
        csv_source = os.path.join(page_dir, "wiserep_spectra.csv")
        csv_dest = os.path.join(datadir, f"wiserep_spectra_page{page}.csv")

        if os.path.exists(csv_source):
            shutil.move(csv_source, csv_dest)
            logger.info(f"Successfully downloaded metadata for page {page}")

            # Now process spectrum files - they should be in the page_dir
            spectrum_files = [
                f
                for f in os.listdir(page_dir)
                if f.lower().endswith(
                    (
                        ".ascii",
                        ".dat",
                        ".txt",
                        ".flm",
                        ".csv",
                        ".asc",
                        ".asci",
                        ".spec",
                        ".text",
                        ".cat",
                        ".mod",
                        ".ecsv",
                        "tns.",
                        "tns_0.",
                        "tns",
                        ".tab",
                        ".spek",
                        ".mod",
                        "salt",
                        "none.",
                        "epesstop.",
                        "clean",
                        "ascii",
                        ".cal",
                        ".0",
                        ".1",
                        ".2",
                        ".3",
                        ".4",
                        ".5",
                        ".6",
                        ".7",
                        ".8",
                        ".9",
                        ".tx",
                        ".",
                        ".xy",
                        "galsub",
                        ".noheader",
                    )
                )
            ]

            if spectrum_files:
                # Create spectra directory if it doesn't exist
                spectra_dir = os.path.join(datadir, "spectra")
                os.makedirs(spectra_dir, exist_ok=True)

                # Move spectrum files to spectra directory
                for spec_file in spectrum_files:
                    source = os.path.join(page_dir, spec_file)
                    dest = os.path.join(spectra_dir, spec_file)
                    if os.path.exists(dest):
                        # Handle duplicate filenames
                        base, ext = os.path.splitext(spec_file)
                        dest = os.path.join(spectra_dir, f"{base}_page{page}{ext}")
                    shutil.move(source, dest)
                logger.info(
                    f"Moved {len(spectrum_files)} spectrum files from page {page}"
                )
            else:
                logger.warning(f"No spectrum files found for page {page}")
        else:
            raise FileNotFoundError(
                f"CSV file not found after extraction for page {page}"
            )

        # Clean up
        os.remove(zip_filename)
        try:
            # Clean up page directory if it's empty
            if len(os.listdir(page_dir)) == 0:
                os.rmdir(page_dir)
        except:
            pass

        # Check if there's actual data in the CSV
        try:
            df = pd.read_csv(csv_dest, index_col=0)
            if len(df) == 0:
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
            f"No valid zip file found for page {page} - possibly no spectra available"
        )
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing page {page}: {e}")
        return False


def download_all_wiserep_data(
    datadir,
    start_page=1,
    max_pages=None,
    obj_type=None,
    redshift_range=None,
    creator=None,
    inst=None,
    tel=None,
    delay=2,
    num_threads=1,
):
    """Download data from all pages with specified filters using multiple threads"""

    logger = setup_logging(datadir)
    session = get_session()

    logger.info(
        f"Starting WISeREP data download with filters: "
        + f"obj_type={obj_type}, redshift_range={redshift_range}, "
        + f"creator={creator}, instrument={inst}, telescope={tel}"
        + f" using {num_threads} threads"
    )

    # Determine the range of pages to download
    if max_pages:
        end_page = start_page + max_pages - 1
    else:
        end_page = float("inf")  # Unlimited pages

    # Function for worker threads
    def download_worker(page_queue, results, lock):
        thread_session = get_session()  # Each thread gets its own session
        while not page_queue.empty():
            try:
                current_page = page_queue.get(block=False)
                if current_page > end_page:
                    page_queue.task_done()
                    continue

                # Add delay between requests from the same thread to avoid hammering the server
                time.sleep(
                    delay * 0.5
                )  # Reduced delay since we're using multiple threads

                success = download_wiserep_data(
                    current_page,
                    datadir,
                    thread_session,
                    logger,
                    obj_type,
                    redshift_range,
                    creator,
                    inst,
                    tel,
                )

                with lock:
                    results[current_page] = success
                    # Update progress bar
                    pbar.update(1)

                page_queue.task_done()
            except Exception as e:
                logger.error(f"Thread error processing page {current_page}: {e}")
                page_queue.task_done()

    # Initialize a queue for pages and a dictionary for results
    page_queue = Queue()
    results = {}
    lock = threading.Lock()

    # Start with a reasonable number of pages to queue initially
    initial_pages = min(100, max_pages or 100)
    for page in range(start_page, start_page + initial_pages):
        page_queue.put(page)
    current_page = start_page + initial_pages

    # Create and start worker threads
    threads = []

    with tqdm(desc="Downloading pages", unit="page") as pbar:
        # Start worker threads
        for _ in range(num_threads):
            thread = threading.Thread(
                target=download_worker, args=(page_queue, results, lock)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # Main thread monitors progress and adds more pages if needed
        empty_pages_sequence = 0
        max_empty_pages = 3  # Stop after this many consecutive empty pages

        while True:
            # Check if we need to add more pages
            if page_queue.qsize() < num_threads and current_page <= end_page:
                page_queue.put(current_page)
                current_page += 1

            # Check if we have consecutive empty pages
            consecutive_empty = 0
            sorted_keys = sorted(results.keys())
            for i in range(
                len(sorted_keys) - 1,
                max(-1, len(sorted_keys) - max_empty_pages - 1),
                -1,
            ):
                if i >= 0 and not results.get(sorted_keys[i], True):
                    consecutive_empty += 1
                else:
                    break

            if consecutive_empty >= max_empty_pages:
                logger.info(
                    f"Detected {max_empty_pages} consecutive empty pages. Stopping."
                )
                break

            # Check if all work is done
            if page_queue.empty() and all(not t.is_alive() for t in threads):
                break

            time.sleep(0.1)  # Small sleep to prevent CPU spinning

    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=5)

    # Count successful pages
    successful_pages = sum(1 for success in results.values() if success)
    logger.info(f"Download complete. Successfully processed {successful_pages} pages.")
    return successful_pages


def verify_and_combine_data(datadir, logger=None):
    """Verify downloaded CSV files and combine them into a single dataset"""

    if logger is None:
        logger = setup_logging(datadir)

    all_files = [
        f
        for f in os.listdir(datadir)
        if f.startswith("wiserep_spectra_page") and f.endswith(".csv")
    ]

    if not all_files:
        logger.warning("No CSV files found to combine")
        return None

    # Sort files by page number
    try:
        all_files.sort(key=lambda f: int(f.split("page")[1].split(".")[0]))
    except (ValueError, IndexError):
        # Fall back to simple sort if we can't extract page numbers
        all_files.sort()

    logger.info(f"Found {len(all_files)} CSV files to combine")

    # Combine all CSVs
    df_list = []
    for file in tqdm(all_files, desc="Combining files"):
        try:
            file_path = os.path.join(datadir, file)
            df = pd.read_csv(file_path, index_col=0)

            # Basic verification
            if df.empty:
                logger.warning(f"File {file} contains no data")
                continue

            df_list.append(df)
            logger.info(f"Added {len(df)} rows from {file}")
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")

    # Combine all dataframes
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        # Remove duplicates if any
        if len(combined_df) > 0:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates()
            after_dedup = len(combined_df)
            if before_dedup > after_dedup:
                logger.info(f"Removed {before_dedup - after_dedup} duplicate entries")

        output_file = os.path.join("wiserep_spectra_combined.csv")
        combined_df.to_csv(output_file, index=False)
        logger.info(
            f"Combined {len(df_list)} files with {len(combined_df)} total rows to {output_file}"
        )

        return combined_df
    else:
        logger.warning("No valid files to combine")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download spectral data from WISeREP")
    parser.add_argument(
        "-d",
        "--datadir",
        default="wiserep_data",
        help="Directory to store downloaded data",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=None,
        help="Starting page number (default: resume from last)",
    )
    parser.add_argument(
        "-m",
        "--max",
        type=int,
        default=None,
        help="Maximum number of pages to download",
    )
    parser.add_argument(
        "-t", "--type", default=None, help="Filter by object type (e.g., 'SN Ia')"
    )
    parser.add_argument(
        "-z",
        "--redshift",
        nargs=2,
        type=float,
        default=None,
        help="Filter by redshift range (min max)",
    )
    parser.add_argument(
        "-c", "--creator", default=None, help="Filter by creator (e.g., 'TNS_Bot1')"
    )
    parser.add_argument("-i", "--instrument", default=None, help="Filter by instrument")
    parser.add_argument("-l", "--telescope", default=None, help="Filter by telescope")
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--threads", type=int, default=10, help="Number of download threads to use"
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only combine existing files, don't download",
    )
    parser.add_argument(
        "--organize-only",
        action="store_true",
        help="Only organize existing files, don't download",
    )

    args = parser.parse_args()

    # Create data directory
    os.makedirs(args.datadir, exist_ok=True)
    logger = setup_logging(args.datadir)

    # Get downloaded, missing, and max page
    downloaded_pages, missing_pages, max_page = find_downloaded_pages(args.datadir)

    if not args.combine_only and not args.organize_only:
        # First handle missing pages if any
        if missing_pages:
            logger.info(
                f"Found {len(missing_pages)} missing pages in sequence: {missing_pages}"
            )
            logger.info("Downloading missing pages first...")

            # Process each missing page
            for page in missing_pages:
                logger.info(f"Downloading missing page {page}")
                download_wiserep_data(
                    page,
                    args.datadir,
                    get_session(),
                    logger,
                    obj_type=args.type,
                    redshift_range=args.redshift,
                    creator=args.creator,
                    inst=args.instrument,
                    tel=args.telescope,
                )
                time.sleep(args.delay)  # Add delay between requests

            logger.info("Finished downloading missing pages")

        # Now determine where to start new downloads
        if args.start is None:
            args.start = max_page + 1
            logger.info(f"Resuming from page {args.start}")

        # Download new pages
        download_all_wiserep_data(
            args.datadir,
            start_page=args.start,
            max_pages=args.max,
            obj_type=args.type,
            redshift_range=args.redshift,
            creator=args.creator,
            inst=args.instrument,
            tel=args.telescope,
            delay=args.delay,
            num_threads=args.threads,
        )

    # Combine data if needed
    if not args.organize_only or not os.path.exists(
        os.path.join("wiserep_spectra_combined.csv")
    ):
        combined_df = verify_and_combine_data(args.datadir, logger)
    else:
        # Load existing combined data
        try:
            combined_path = os.path.join("wiserep_spectra_combined.csv")
            combined_df = pd.read_csv(combined_path, index_col=0)
            logger.info(f"Using existing combined data with {len(combined_df)} rows")
        except Exception as e:
            logger.error(f"Error loading existing combined data: {e}")
            combined_df = verify_and_combine_data(args.datadir, logger)

    logger.info("Processing complete")


if __name__ == "__main__":
    main()
