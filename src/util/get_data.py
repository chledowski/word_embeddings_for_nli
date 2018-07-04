import zipfile
import os
from src import DATA_DIR
import urllib.request, urllib.parse, urllib.error


def get_data(url, folder):
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'whatever')
    filename, headers = opener.retrieve(url, 'Test.pdf')
    urllib.request.urlretrieve(url, os.path.join(DATA_DIR, "data.zip"))
    zip_ref = zipfile.ZipFile(os.path.join(DATA_DIR, 'data.zip'), 'r')
    if not os.path.exists(os.path.join(DATA_DIR, folder)):
        os.makedirs(os.path.join(DATA_DIR, folder))
    zip_ref.extractall(os.path.join(DATA_DIR, folder))
    zip_ref.close()
    os.remove(os.path.join(DATA_DIR, 'data.zip'))
