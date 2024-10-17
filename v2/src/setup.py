import os

def setup(domain):
    if not os.path.exists(f'/Volumes/T9/domains/{domain}'):
        os.makedirs(f'/Volumes/T9/domains/{domain}')