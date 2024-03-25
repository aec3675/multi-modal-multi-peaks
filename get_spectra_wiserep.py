# Snippet to download all TNS spectra via WISeRep

import os
import numpy as np
from urllib.request import urlretrieve


for page in np.arange(1,1000):
    url = f"https://www.wiserep.org/search/spectra?&creation_modifier=TNS_Bot1&format=csv&files_type=ascii&num_page=100&page={int(page)}"
    print(url)
    try:
        os.system(f"""wget -O test_{page}.zip "{url}" --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" -nc  """)
        print(f"""wget -O test_{page}.zip {url} --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" """)
        os.system(f"unzip test_{page}.zip")
    except:
        print(f"No spectra in page {page}")
    os.system(f"mv wiserep_spectra.csv wiserep_spectra_page{page}.csv")
