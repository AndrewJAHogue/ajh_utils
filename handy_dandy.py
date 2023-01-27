
# %%
import os
from re import M
import sys
import joblib

import numpy as np
from astropy.nddata import Cutout2D
# sys.path.append('F:/Python/modules')
# from modules.ajh_utils import lineplots, computer_path
from astropy.io import fits

# %%

def get_region_cutouts( x, y, h, w,):
    """Loads the SOFIA and Spitzer datasets and creates a cutout centered at (x,y) of size (h, w) 

    Args:
        x (int): x coordinate to center cutout on
        y (int): y coordinate to center cutout on
        h (int): heighth of returned cutouts
        w (int): width of returned cutouts

    Returns:
        Returns two cutouts of equal size, centered on the same (x, y). Returns a cutout from the SOFIA dataset and a cutout from the Spitzer dataset; tuple-like
    """    

    fits_dir = '../Research/fits/Full Maps/Originals/'
    # spits_data = computer_path.Star_Datasets.get_spits_data()
    # sofia_data = computer_path.Star_Datasets.get_sofia_data()
    spits_data = fits.getdata(f'{fits_dir}Spitzer_GCmosaic_24um_onFORCASTheader_JyPix.fits')
    sofia_data = fits.getdata(f'{fits_dir}F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits')

    spits_cutout = Cutout2D(spits_data, (x, y), (h, w)).data
    sofia_cutout = Cutout2D(sofia_data, (x, y), (h, w)).data

    return sofia_cutout, spits_cutout

    
def rms(y_true, y_pred):
    """Calculate rms error of predictions

    Args:
        y_true (array-like): predictions
        y_pred (array-like): testing data

    Returns:
        ndarray  
    """    

    from numpy import sqrt
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y_true, y_pred)
    return sqrt(mse) 


def getSourcesList(input_data, sigma=3.0, fwhm=10., threshold=5.):
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    
    mean, median, std = sigma_clipped_stats(input_data, sigma=sigma, stdfunc=np.nanstd)
    d = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    s = d(input_data - median)
    for col in s.colnames:
        s[col].info.format = "%.8g"

    vals = s['flux']
    mask = vals > std
    s.remove_rows(mask)
    
    print(f'Found {len(s)} stars')

    return s




# from photutils.aperture import CircularAperture
# from photutils.detection import DAOStarFinder
# from astropy.visualization.mpl_normalize import ImageNormalize
# from astropy.visualization import SqrtStretch
# from  photutils.aperture import ApertureStats
# positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
# apertures = CircularAperture(positions, r=10.)
# norm = ImageNormalize(stretch=SqrtStretch())

def createCutoutsList(input_data,  **keywargs):
    """_summary_

    Args:
        input_data (2D array): the data of a fits star file
        save_fwhm (boolean)
        cutout_size (tuple of two)
        threshold (float)
        sigma (float)
        fwhm (float)
        auto_filter (boolean)
        filename (string)

    Returns:
        (Cutouts, headers)
    """
    save_fwhm = keywargs.get('save_fwhm', False)
    cutout_size = keywargs.get('cutout_size', (50, 50))
    threshold = keywargs.get('threshold', 5.)
    sigma = keywargs.get('sigma', 3.)
    fwhm = keywargs.get('fwhm', 10.)
    do_filter = keywargs.get('do_filter', False)
    filename = keywargs.get('filename', 'createCutoutsList_Filler_Filename')
    
    from astropy.stats import sigma_clipped_stats
    
    sources_list = getSourcesList(input_data, sigma=sigma, fwhm=fwhm, threshold=threshold)
    stats = sigma_clipped_stats(input_data, sigma=sigma, stdfunc=np.nanstd)


    # x = sources_list["xcentroid"]
    # y = sources_list["ycentroid"]
    # points = list(zip(x, y))
    cutouts = []
    masked_cutouts = []
    cutouts_headers = []


    for source in sources_list:
        point = ( source['xcentroid'], source['ycentroid'])
        c = Cutout2D(input_data, point, cutout_size)
        
        ## a filter for any point source that exceeds the std of the mosaic
        mean = np.nanmean(c.data)
        if mean > (stats[0] + stats[2]) and do_filter:
            continue 
        if c.shape != cutout_size:
            continue


        # save fwhm of original cutout
        if save_fwhm:
            try:
                fwhm = get_fwhm(c.data)
            except ValueError:
                fwhm = "error"
            
            save_fwhm_to_file(fwhm, point, filename)

        cutouts.append(c.data)
        cutouts_headers.append(c)

    cutouts = np.array(cutouts)
    cutouts_headers = np.array(cutouts_headers)

    print(f'Created List of Cutouts with size of {len(cutouts)}')
    
    return cutouts, cutouts_headers

def createMaskedCutoutsList(input_data, **keywargs):
    sigma = keywargs.get('sigma', 3.)
    nsigma = keywargs.get('nsigma', 10.)
    radius = keywargs.get('radius', 8.)
    fwhm = keywargs.get('fwhm', 10.)
    threshold = keywargs.get('threshold', 5.)
    auto_filter = keywargs.get('filter', False)
    peak_percentage = keywargs.get('peak_percentage', 0.7)

    from astropy.stats import sigma_clipped_stats, SigmaClip
    from photutils.detection import DAOStarFinder

    sources = getSourcesList(input_data, sigma, fwhm, threshold)
    stats = sigma_clipped_stats(input_data, sigma=sigma, stdfunc=np.nanstd)

    ## go to each source and make a cutout
    ## training are cutouts that have their peaks artifically masked
    ## testing cutouts are the original cutouts
    training_cutouts = []
    testing_cutouts = []
    headers_cutouts = []
    mean = np.nanmean(input_data)
    for s in sources:
        x = s['xcentroid']
        y = s['ycentroid']
        
        header = Cutout2D(input_data, (x, y), 50)
        testing = Cutout2D(input_data, (x, y), 50).data

        ## ignore the edge cutouts that end up having sizes smaller than (50, 50)
        if testing.shape != (50, 50):
            continue

        # ## skip the cutout that is outside the mean + std fro the whole input_data
        # if mean > (stats[0] + stats[2]) and auto_filter:
        #     continue 
        
        ## mask the peak and then some using a configurable coefficient
        peak = np.nanmax(testing)
        ## .70 is artifically chosen
        ## mask only 30% of the peak flux
        cutoff = peak * peak_percentage

        masked = np.copy(testing)
        if masked.shape != (50, 50):
            continue

        mask = masked > cutoff
        masked[mask] = np.NaN

        ## use the masked data
        training_cutouts.append(masked)
        testing_cutouts.append(testing)
        headers_cutouts.append(header)

    return training_cutouts, testing_cutouts, headers_cutouts

    
    

def getFileName(input_str):
    try:
        dot = input_str.rindex('.')
        return input_str[:dot]
    except ValueError as exc:
        raise ValueError("This string does not contain an extension substring") from exc

def saveCutoutsHeaders(cutouts_headers, filename):
    """Take in a tuple of (cutouts, headers) and a string filename and saves it as a joblib

    Args:
        cutouts_headers (tuple of (ndarray, ndarray)): tuple of cutouts and headers
        filename (string): full filename. The filename needs to have 
    Returns:
        returns nothing
    """    

    output = f'{getFileName(filename)}_cutouts_headers.jbl'
    with open(f'./datasets/cutouts/{output}', 'wb') as f:
        joblib.dump(cutouts_headers, f)

CUTOUT_SIZE = 50
def maskBackground(input_data, CUTOUT_SIZE=CUTOUT_SIZE, sigma=6.0):
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    sigma_clip = SigmaClip(sigma=sigma)

    bkg_estimator = MedianBackground()

    bkg = Background2D(
        input_data, CUTOUT_SIZE, filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator
    )

    print(f"{bkg.background_median = }\n{bkg.background_rms_median = }")

    return input_data - bkg.background


def mask_sources(input_data, sigma, nsigma, radius = 8):
    """_summary_

    Args:
        input_data (int): _description_
        sigma (int): _description_
        nsigma (int): _description_
        radius (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """    
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import detect_sources, detect_threshold
    from photutils.utils import circular_footprint

    sigma_clip = SigmaClip(sigma=sigma, maxiters=10)
    threshold = detect_threshold(input_data, nsigma=nsigma, sigma_clip=sigma_clip)
    segment_img = detect_sources(input_data, threshold, npixels=5)
    footprint = circular_footprint(radius=radius)
    
    if segment_img:
        mask = segment_img.make_source_mask(footprint=footprint)
    else:
        raise AttributeError("No sources found, with the selected parameters")
    copy_data = np.copy(input_data)
    copy_data[mask] = np.NaN
    return copy_data


def processData(input_data, sigma=3.):
    """
    Description:
        Mask the background noise and impute any nan values 
    Arguments:
        input_data: Must be a 2D array
    Returns:
        Numpy array
    """    



    # masking the background
    masked = maskBackground(input_data, CUTOUT_SIZE, sigma)

    # imputed nan values
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(missing_values=np.NaN, n_neighbors=40)
    return imputer.fit_transform(masked)



def calc_fwhm(cutout_data):
    import numpy as np
    from scipy.interpolate import CubicSpline, PPoly, UnivariateSpline, splrep
    from specutils.analysis import gaussian_fwhm

    # cutout_data = Cutout2D(data, star_coord, cutout_size).data
    cutout_size = cutout_data.shape[0]
    
    light_profile = np.take(cutout_data, int(cutout_size / 2), axis=0)
    half_max = np.nanmax(cutout_data) / 2
    x = np.linspace(0, len(light_profile), len(light_profile))
    spline = UnivariateSpline(x, light_profile - half_max, s=0)
    # spline = splrep(x, light_profile - half_max, s=0)
    # p = PPoly.from_spline(spline)
    r1, r2 = spline.roots()
    fwhm = r2 - r1
    return fwhm

def save_fwhm_to_file(fwhm, point, filename):
    # import json
    from os.path import isfile

    import joblib

    if '.fits' in filename:
        filename = filename[:len(filename) - 5]

    FILE = f'./datasets/fwhms/{filename}_fwhm.joblib'

    
    # check if file exists
    if isfile(FILE) == False:
        with open(FILE, 'xb') as f:
            pass
        
    fwhm_list = list([ {'fwhm': fwhm, 'coordinate': point } ])

    # print(f'{fwhm_list = }')
    
    with open(FILE, 'wb') as f:
        joblib.dump(fwhm_list, f)    



def twoD_GaussianScaledAmp(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
# Compute FWHM(x,y) using 2D Gaussian fit, min-square optimization
# Optimization fits 2D gaussian: center, sigmas, baseline and amplitude
# works best if there is only one blob and it is close to the image center.
# author: Nikita Vladimirov @nvladimus (2018).
# based on code example: https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m

    import numpy as np
    (x, y) = xy
    xo = float(xo)
    yo = float(yo)    
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()

def getFWHM_GaussianFitScaledAmp(img):
    """Get FWHM(x,y) of a blob by 2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns: 
        FWHMs in pixels, along x and y axes.
    """
    import numpy as np
    import scipy.optimize as opt
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    initial_guess = (img.shape[1]/2,img.shape[0]/2,10,10,1,0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, ( x, y ), 
                               img_scaled.ravel(), p0=initial_guess,
                               bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -0.1),
                               (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5)))
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return (FWHM_x, FWHM_y)

def get_fwhm(cutout):
    try:
        fwhm_maj, fwhm_min = getFWHM_GaussianFitScaledAmp(cutout)
        return np.sqrt(fwhm_maj**2 + fwhm_min**2)
    except RuntimeError:
        if np.any(np.isnan(cutout)):
            print('WARNING: This cutout contains invalid values')

def saveFWHMFile(data, filename):
    import joblib
    with open(f'./datasets/{filename}.fits', 'wb') as f:
        joblib.dump(data, f)

        
def FilterStarsByStd(cutouts, stats):
    filtered_cutouts = []
    for c in cutouts:
        mean = np.nanmean(c)
        if mean < (stats[0] + stats[2]):
            filtered_cutouts.append(c)
        
    # if len(filtered_cutouts) == 1:
    #     filtered_cutouts = filtered_cutouts[0]

    print(f'Filtered down to {len(filtered_cutouts)} stars')
    
    return filtered_cutouts

# %%
def main():
    from astropy.io import fits
    ## -------FILETREE-------------
    ## -datasets
    ## --MG
    ## --- all the MG fits files
    ##
    ## -modules
    ## --handy_dandy
    ## ---current file
    mg610p005 = fits.getdata('../../datasets/MG/MG0610p005_024.fits')

    createCutoutsList(mg610p005)
if __name__ == "__main__":
    main()

