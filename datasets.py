import shutil
import urllib
import requests
import os
from tqdm import tqdm
import urllib.request as req
import numpy as np
import ftplib
import shutil

def downloadVision(downloadPath) : 
    downloadPath = os.path.join(downloadPath, "VISION")
    
    if not os.path.exists(downloadPath) : 
        os.makedirs(downloadPath)

    BASE_FILE_URL = "https://lesc.dinfo.unifi.it/VISION/VISION_base_files.txt"  # 각 이미지의 download url이 작성된 파일의 다운로드 URL
    BASE_FILE_PATH = os.path.join(downloadPath, "VISION_base_files.txt")  # BASE FILE을 저장할 경로
    download(BASE_FILE_URL, BASE_FILE_PATH)  # BASE FILE 다운로드

    with open(BASE_FILE_PATH, "rt") as basefile : 
        urls = basefile.readlines()

        for url in tqdm(urls): 
            if "video" in url : 
                continue

            img_path = os.path.join(downloadPath, url.rsplit("dataset/", 1)[1].replace("\n", ""))
            if not os.path.exists(img_path) : 
                download(url, img_path)

def downloadSOCRatES(downloadPath) : 
    downloadPath = os.path.join(downloadPath, "SOCRatES")
    LAST_PART = 11

    for part in tqdm(range(1, LAST_PART + 1)) : 
        filename = "SOCRatES_part{}.zip".format(part)
        download("ftp://ftp.eurecom.fr/incoming/{}".format(filename), os.path.join(downloadPath, filename))

def download(url, filepath) : 
    dirpath = os.path.split(filepath)[0]

    if not os.path.exists(dirpath) : 
        os.makedirs(dirpath)

    req.urlretrieve(url, filepath)

if __name__ == "__main__" : 
    # downloadVISION("C:/Users/rlfalsgh95/source/repos/Source_Identification_Datasets")

