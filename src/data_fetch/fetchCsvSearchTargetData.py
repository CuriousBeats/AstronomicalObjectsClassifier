import requests as rqObj
import pandas as pd

from configs.sdssApiEndpoints import SDSS_IMAGE_CUTOUT_BASE, SDSS_SPECTRA_BASE, SDSS_OBJ_SQL_SEARCH_BASE

def initDataSizeFetch():
    initSqlGetResultCount = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT COUNT(*) FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid&format=json"
    sizeResp = rqObj.get(initSqlGetResultCount)
    if(sizeResp.status_code == 200):
        data = sizeResp.json()
        initDataSize = data[0]['Rows'][0]['Column1']
    
    return initDataSize

def remDataSizeFetch(lastObjId):
    sqlGetResultCount = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT COUNT(*) FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE p.objid > " + str(lastObjId) + "&format=json"
    sizeResp = rqObj.get(sqlGetResultCount)
    if(sizeResp.status_code == 200):
        data = sizeResp.json()
        dataSize = data[0]['Rows'][0]['Column1']
    
    return dataSize

def multiPartFileWriter(resp, csvFilePath):
    with open(csvFilePath, 'wb') as file:
        file.write(resp.content)

def getLastObjId(lastBatchFile):
    df = pd.read_csv(lastBatchFile, comment='#')
    last_objid = df['objid'].iloc[-1]
    return last_objid

def fetchCsvMultiBatchData():
    iter = 1
    size = initDataSizeFetch()
    getAstroDataInit = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT p.objid, s.specobjid, s.class, p.ra as ra, p.dec as dec, p.run AS r, p.camcol AS c, p.field as field FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid ORDER BY p.objid&format=csv"
    initResponse = rqObj.get(getAstroDataInit)
    multiPartFileWriter(initResponse, 'data/raw/csv_extract/astro_data_batch_' + str(iter) + '.csv')
    lastObjectID = getLastObjId('data/raw/csv_extract/astro_data_batch_' + str(iter) + '.csv')

    while(size != 0):
        iter += 1
        getAstroDataBatched = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT p.objid, s.specobjid, s.class, p.ra as ra, p.dec as dec, p.run AS r, p.camcol AS c, p.field as field FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE p.objid > " + str(lastObjectID) + " ORDER BY p.objid&format=csv"
        batchedDataResponse = rqObj.get(getAstroDataBatched)
        batchFileNamePath = 'data/raw/csv_extract/astro_data_batch_' + str(iter) + '.csv'
        multiPartFileWriter(batchedDataResponse, batchFileNamePath)
        lastObjectID = getLastObjId(batchFileNamePath)
        size = remDataSizeFetch(lastObjectID)
        print("\nSize: " + str(size) + "\nLast Object ID: " + str(lastObjectID))

        
fetchCsvMultiBatchData()