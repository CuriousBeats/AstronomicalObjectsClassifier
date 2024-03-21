import requests as rqObj

from configs.sdssApiEndpoints import SDSS_IMAGE_CUTOUT_BASE, SDSS_SPECTRA_BASE, SDSS_OBJ_SQL_SEARCH_BASE

def testDataFetch(url):
    '''
    Debug tests

    print("Image base URL: " + SDSS_IMAGE_CUTOUT_BASE)
    print("Spectra base URL: " + SDSS_SPECTRA_BASE)
    print("SQL Search URL: " + SDSS_OBJ_SQL_SEARCH_BASE)
    '''
    #print(url)
    response = rqObj.get(url)
    print(response.json())
    

sqlSearchUrl = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT TOP 10 p.objid, s.specobjid, s.class, p.ra as ra, p.dec as dec, p.run AS r, p.camcol AS c, p.field as field FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid&format=json"
spectraSearchUrl = SDSS_SPECTRA_BASE + "id="
testDataFetch(sqlSearchUrl)
