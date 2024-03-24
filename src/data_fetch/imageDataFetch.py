import requests as reqObj
import pandas as pd
from configs.sdssApiEndpoints import SDSS_IMAGE_CUTOUT_BASE, SDSS_SPECTRA_BASE

def fetchImageData(csvFilePath):
    print("Reading Data File: " + str(csvFilePath))
    df = pd.read_csv(csvFilePath, comment='#')
    for index, row in df.iterrows():
        ra = row['ra']
        dec = row['dec']
        objid = row['objid']
        specobjid = row['specobjid']
        imageClass = row['class']
        print(f"Row {index+1}: RA = {ra}, DEC = {dec}, OBJID = {objid}, SPEC_OBJID = {specobjid}, IMAGE_CLASS = {imageClass}")
        getImage(objid, specobjid, imageClass, ra, dec, 0.4, 128, 128, "")

def writeImageFile(resp, imgFilePath):
    print("Writing " + str(imgFilePath) + " ....")
    with open(imgFilePath, 'wb') as file:
        file.write(resp.content)
        print("Written " + str(imgFilePath) + " ....\n")

def getImage(objId, specObjId, imgClass, ra, dec, scale, height, width, opt):

    galaxyImagePath = "data/raw/image_extracts/astroImages/galaxy/"
    starImagePath = "data/raw/image_extracts/astroImages/star/"
    qsoImagePath = "data/raw/image_extracts/astroImages/qso/"

    galaxySpecImagePath = "data/raw/image_extracts/specImages/galaxy/"
    starSpecImagePath = "data/raw/image_extracts/specImages/star/"
    qsoSpecImagePath = "data/raw/image_extracts/specImages/qso/"


    if(opt is None):
        searchUrl = "ra=" + str(ra) + "&dec=" + str(dec) + "&scale=" + str(scale) + "&height=" + str(height) + "&width=" + str(width)
    else:
        searchUrl = "ra=" + str(ra) + "&dec=" + str(dec) + "&scale=" + str(scale) + "&height=" + str(height) + "&width=" + str(width) + "&opt=" + str(opt)

    imageResult = reqObj.get(SDSS_IMAGE_CUTOUT_BASE + searchUrl)
    
    if(imgClass == "GALAXY"):
        imagePath = galaxyImagePath + str(objId) + ".png"
    elif(imgClass == "STAR"):
        imagePath = starImagePath + str(objId) + ".png"
    elif(imgClass == "QSO"):
        imagePath = qsoImagePath + str(objId) + ".png"
    else:
        print("Unknown or empty class: " + str(imgClass) + " ID: " + str(objId))

    writeImageFile(imageResult, imagePath)

fetchImageData("data/raw/csv_extract/astro_data_batch_1.csv")
