# Snippet to download all TNS spectra via WISeRep

import asyncio
import aiohttp
import logging
from pathlib import Path
from zipfile import ZipFile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
URL_TEMPLATE = "https://www.wiserep.org/search/spectra?&creation_modifier=TNS_Bot1&format=csv&files_type=ascii&num_page=100&page={page}"
CONCURRENT_REQUESTS = 5

async def download_file(session, url, zip_filename, semaphore):
    async with semaphore:
        async with session.get(url) as response:
            response.raise_for_status()
            with open(zip_filename, 'wb') as f:
                f.write(await response.read())

async def download_and_process(session, start_page, end_page, semaphore):
    for page in range(start_page, end_page):
        url = URL_TEMPLATE.format(page=int(page))
        logging.info(f"Processing URL: {url}")
        try:
            zip_filename = f"test_{page}.zip"
            await download_file(session, url, zip_filename, semaphore)

            # Unzip the downloaded file
            with ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall()

            # Move the file if it exists
            csv_path = Path("wiserep_spectra.csv")
            if csv_path.exists():
                csv_path.rename(f"wiserep_spectra_page{page}.csv")

            # Clean up the zip file
            Path(zip_filename).unlink()

        except Exception as e:
            logging.error(f"Error on page {page}: {str(e)}")

async def main():
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
        tasks = [download_and_process(session, i, i + 100, semaphore) for i in range(1, 1000, 100)]
        await asyncio.gather(*tasks)

# Run the main function
asyncio.run(main())