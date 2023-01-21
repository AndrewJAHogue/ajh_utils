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
