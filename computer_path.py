import json
from astropy.io import fits 
from sys import platform

#LIB_PATH_WIN = 'C:/Users/ahogue5/Documents/anaconda3/Lib/site-packages/ajh_utils'

paths = {
    "win32": "F:",
    "linux": "/media/al-linux/USB20FD/"
}
def get_computer_path():
    return paths[platform]

def FitsFolder():
    return f'{get_computer_path()}/Python/Research/fits'


def spits_iso1():
    return  f'{FitsFolder()}/Reprojected Spitzer24 IsoFields @ Forcast25 isoField1.fits'

def spits_iso2():
    return f'{FitsFolder()}/Reprojected Spitzer24 IsoFields @ Forcast25 isoField2.fits'

def spits_sgrb():
    return f'{FitsFolder()}/Reprojected Spitzer24_SgrB @ Forcast25_SgrB.fits'

def sofia_iso1():
    return  f'{FitsFolder()}/Forcast25_isoField1.fits'
    
def spits_iso2():
    return  f'{FitsFolder()}/Forcast25_isoField2.fits'

def spits_sgrb():
    return f'{FitsFolder()}/Forcast25_SgrB.fits'

class FullMaps():
    def Sofia():
        return f'{FitsFolder()}/Full Maps/Originals/F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits'

    def Spitzer():
        return f'{FitsFolder()}/Full Maps/Originals/Spitzer_GCmosaic_24um_onFORCASTheader_JyPix.fits'

class Star_Datasets():
    def get_spits_data():
        return fits.getdata(FullMaps.Spitzer())
        # return joblib.load("./datasets/spits_data.joblib")

    def get_sofia_data():
        # return joblib.load("./datasets/spits_data.joblib")
        return fits.getdata(FullMaps.Sofia())
