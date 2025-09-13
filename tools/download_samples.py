import os
from urllib.request import urlretrieve

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples')

URLS = [
    # Public sample DXF (MIT licensed repo)
    'https://raw.githubusercontent.com/jscad/sample-files/master/dxf/dxf-parser/floorplan.dxf',
]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def filename_from_url(url: str) -> str:
    return url.split('/')[-1]

def download_files():
    ensure_dir(SAMPLES_DIR)
    downloaded = []
    for url in URLS:
        fname = filename_from_url(url)
        dest = os.path.join(SAMPLES_DIR, fname)
        if not os.path.exists(dest):
            urlretrieve(url, dest)
        downloaded.append(dest)
    return downloaded

if __name__ == '__main__':
    paths = download_files()
    print('\n'.join(paths))


