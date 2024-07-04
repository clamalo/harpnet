import sys, os
import requests
import tarfile
import xarray as xr
import os
import numpy as np
from calendar import monthrange
import dask
from dask.diagnostics import ProgressBar
import warnings
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
warnings.filterwarnings("ignore")
from tqdm import tqdm
import re


dspath = 'https://request.rda.ucar.edu/dsrqst/MALOTT748122/'

def check_file_status(filepath):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size_mb = int(os.stat(filepath).st_size)/ 1000**2
    sys.stdout.write('Downloaded %.1f MB' % (size_mb,))
    sys.stdout.flush()

with open('/Volumes/T9/filelist.txt', 'r') as filehandle:
    filelist = [line.strip() for line in filehandle.readlines()]
    print(len(filelist), 'files to download')

    tar_path = '/Volumes/T9/tar/'
    nc_dir = '/Volumes/T9/nc/'
    for file in filelist:
        filename=dspath+file


        start_year = filename[87:91]


        file_base = os.path.basename(file)

        # get rid of the first part when you do .split('.')
        file_base = file_base.split('.')[1:]
        file_base = '.'.join(file_base)

        print('Downloading', file_base)

        req = requests.get(filename, allow_redirects=True, stream=True)
        with open(os.path.join(tar_path, file_base), 'wb') as outfile:
            chunk_size = 10485760
            for chunk in req.iter_content(chunk_size=chunk_size):
                outfile.write(chunk)
                check_file_status(os.path.join(tar_path, file_base))

        with tarfile.open(os.path.join(tar_path, file_base)) as tar:
            for member in tqdm(tar.getmembers()):
                member.name = member.name.split('.')[1:]
                member.name = '.'.join(member.name)
                tar.extract(member, path=nc_dir)

        os.remove(os.path.join(tar_path, file_base))
        print('Extracted', file_base)
        print()
    
        #remove the file from the list, write back to file
        filelist = filelist[1:]
        with open('/Volumes/T9/filelist.txt', 'w') as filehandle:
            for listitem in filelist:
                filehandle.write('%s\n' % listitem)