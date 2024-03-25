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
        getImage(objid, specobjid, imageClass, ra, dec, 0.4, 128, 128, "OBFQ")
        if index == 75999:
            break

def writeImageFile(resp, imgFilePath):
    print("Writing " + str(imgFilePath) + " ....")
    with open(imgFilePath, 'wb') as file:
        file.write(resp.content)
        print("Written " + str(imgFilePath) + " ....\n")

def getImage(objId, specObjId, imgClass, ra, dec, scale, height, width, opt):

    galaxyImagePath = "data/raw/image_extracts/astroImages/galaxy/"
    starImagePath = "data/raw/image_extracts/astroImages/star/"
    qsoImagePath = "data/raw/image_extracts/astroImages/qso/"

    invGalaxyImagePath = "data/raw/image_extracts/filteredImages/invFilter/galaxy/"
    invStarImagePath = "data/raw/image_extracts/filteredImages/invFilter/star/"
    invQsoImagePath = "data/raw/image_extracts/filteredImages/invFilter/qso/"

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> upstream/main
    customGalaxyImagePath = "data/raw/image_extracts/filteredImages/OBFQFilter/galaxy/"
    customStarImagePath = "data/raw/image_extracts/filteredImages/OBFQFilter/star/"
    customQsoImagePath = "data/raw/image_extracts/filteredImages/OBFQFilter/qso/"

<<<<<<< HEAD
>>>>>>> upstream/main
=======
>>>>>>> upstream/main
    galaxySpecImagePath = "data/raw/image_extracts/specImages/galaxy/"
    starSpecImagePath = "data/raw/image_extracts/specImages/star/"
    qsoSpecImagePath = "data/raw/image_extracts/specImages/qso/"

    specSearchUrl = SDSS_SPECTRA_BASE + str(specObjId)

    if(opt is None):
        searchUrl = "ra=" + str(ra) + "&dec=" + str(dec) + "&scale=" + str(scale) + "&height=" + str(height) + "&width=" + str(width)
    else:
        searchUrl = "ra=" + str(ra) + "&dec=" + str(dec) + "&scale=" + str(scale) + "&height=" + str(height) + "&width=" + str(width) + "&opt=" + str(opt)

    imageResult = reqObj.get(SDSS_IMAGE_CUTOUT_BASE + searchUrl)
    #specImageResult = reqObj.get(SDSS_SPECTRA_BASE + str(specObjId))
    
    if(imgClass == "GALAXY"):
        imagePath = galaxyImagePath + str(objId) + ".png"
        specImagePath = galaxySpecImagePath + str(objId) + "_spec.png"
        invImagePath = invGalaxyImagePath + str(objId) + "_inv.png"
<<<<<<< HEAD
<<<<<<< HEAD
=======
        customImagePath = customGalaxyImagePath + str(objId) + "_obfq.png"
>>>>>>> upstream/main
=======
        customImagePath = customGalaxyImagePath + str(objId) + "_obfq.png"
>>>>>>> upstream/main
        
    elif(imgClass == "STAR"):
        imagePath = starImagePath + str(objId) + ".png"
        specImagePath = starSpecImagePath + str(objId) + "_spec.png"
        invImagePath = invStarImagePath + str(objId) + "_inv.png"
<<<<<<< HEAD
<<<<<<< HEAD
=======
        customImagePath = customStarImagePath + str(objId) + "_obfq.png"
>>>>>>> upstream/main
=======
        customImagePath = customStarImagePath + str(objId) + "_obfq.png"
>>>>>>> upstream/main
         
    elif(imgClass == "QSO"):
        imagePath = qsoImagePath + str(objId) + ".png"
        specImagePath = qsoSpecImagePath + str(objId) + "_spec.png"
        invImagePath = invQsoImagePath + str(objId) + "_inv.png"
<<<<<<< HEAD
<<<<<<< HEAD
=======
        customImagePath = customQsoImagePath + str(objId) + "_obfq.png"
>>>>>>> upstream/main
=======
        customImagePath = customQsoImagePath + str(objId) + "_obfq.png"
>>>>>>> upstream/main

    else:
        print("Unknown or empty class: " + str(imgClass) + " ID: " + str(objId))

    if(opt is None):
        writeImageFile(imageResult, imagePath)
        #writeImageFile(specImageResult, specImagePath)
    else:
<<<<<<< HEAD
<<<<<<< HEAD
        writeImageFile(imageResult, invImagePath)
=======
        writeImageFile(imageResult, customImagePath)
>>>>>>> upstream/main
=======
        writeImageFile(imageResult, customImagePath)
>>>>>>> upstream/main
        #writeImageFile(specImageResult, specImagePath)

fetchImageData("data/raw/csv_extract/astro_data_batch_1.csv")
