import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Galactic



def GetNthColumn(file, xvalue, **kwargs):
    xmin = kwargs.get('xmin', None)
    xmax = kwargs.get('xmax', None)

    ymin = kwargs.get('ymin', None)
    ymax = kwargs.get('ymax', None)

    fake = kwargs.get('fake', True)

    yvalues = np.array([])
    yvalues = file[:,xvalue]
    xdata = np.array(range(len(yvalues)))

    if min is not None and max is not None:
        yvalues = yvalues[xmin:xmax]
        xdata = xdata[xmin:xmax]
    # fake x data
    # if fake:
    return np.array([xdata, yvalues])
    # else:
        # return np.array(yvalues)

def GetNthRow(file, yvalue, **kwargs):
    xmin = kwargs.get('xmin', None)
    xmax = kwargs.get('xmax', None)

    ymin = kwargs.get('ymin', None)
    ymax = kwargs.get('ymax', None)

    fake = kwargs.get('fake', True)

    xvalues = np.array([])
    xvalues = file[yvalue]
    ydata = np.array(range(len(xvalues)))

    if min is not None and max is not None:
        xvalues = xvalues[xmin:xmax]
        ydata = ydata[xmin:xmax]
    return np.array([ydata, xvalues])

def SingleLinePlot(xvalue, yvalue, columnmin=0.0, rowmin=0.0, **kwargs):
    filepath = kwargs.get('filepath', None) 
    data = kwargs.get('data', None)
    if filepath and ~data:
        data = fits.open(filepath)[0].data
    # if data and filepath != '':
    #     print('You have defined two different arguments as target datasets. Choose "data" or "filepath," not both.')
    #     return



    column = GetNthColumn(data, xvalue)
    columnLineplot = plt.subplot(1,2,1)
    plt.title(f'Column-Pixel Saturation at X={xvalue}')
    plt.xlabel('Y Index')
    plt.ylabel('Pixel Value')
    plt.plot(column[0],column[1])

    row = GetNthRow(data, yvalue)
    rowLineplot = plt.subplot(1,2,2)
    plt.title(f'Row-Pixel Saturation at Y={yvalue}')
    plt.xlabel('X Index')
    plt.ylabel('Pixel Value')
    plt.plot(row[0],row[1])
    plt.suptitle(filepath)

    plt.show()

def MultiLinePlot(xvalue, yvalue, fileset=[], columnlimits=[None,None,None,None], rowlimits=[None,None,None,None], **kwargs):
    # plt.rcParams.update({'font.size': 27})
    colxmin = columnlimits[0]
    colxmax = columnlimits[1]
    colymin = columnlimits[2]
    colymax = columnlimits[3]
    rowxmin = rowlimits[0]
    rowxmax = rowlimits[1]
    rowymin = rowlimits[2]
    rowymax = rowlimits[3]

    
    x2 = kwargs.get('x2', None)
    y2 = kwargs.get('y2', None)

    legend = kwargs.get('legend', False) ## show legend or not
    
    files = fileset.copy()
    datasets = np.array([])
    for arg in kwargs:
        if 'file' in str(arg):
            files = np.append(files, kwargs.get(arg, None))
    grid = 1
    print(f'grid is => {grid}')

    from math import ceil, floor

    for plot_index,file in enumerate(files):
        if isinstance(file, str):
            data = fits.open(file)[0].data
        else:
            data = file
        
        try:
            file = file[file.find('fits/') + 5:] #test[test.find('fits/') + 5:]
            print(file)
        except:
            pass

        column = GetNthColumn(data, xvalue)
        ax1 = plt.subplot(grid,2,1)
        plt.title(f'Column-Pixel Saturation at X={xvalue}')
        plt.xlabel('Y Index')
        plt.ylabel('Pixel Value')
        ax1.margins(0)
        if colxmin != None:
            plt.xlim(left=colxmin)
        if colxmax != None:
            plt.xlim(right=colxmax)
        if colymin != None:
            plt.ylim(bottom=colymin)
        if colymax != None:
            plt.ylim(top=colymax)

        plt.plot(column[0], column[1], label=file)
        if legend:
            plt.legend()

        row = GetNthRow(data, yvalue)
        plt.subplot(grid,2,2)
        plt.title(f'Row-Pixel Saturation at Y={yvalue}')
        plt.xlabel('X Index')
        if rowxmin != None:
            plt.xlim(left=rowxmin)
        if rowxmax != None:
            plt.xlim(right=rowxmax)
        if rowymin != None:
            plt.ylim(bottom=rowymin)
        if rowymax != None:
            plt.ylim(top=rowymax)
        plt.plot(row[0],row[1], label=file) 
        if legend:
            plt.legend()
    plt.show()

def plot_gallery(images, h, w, n_row=3, n_col=4):
    from astropy.nddata import Cutout2D
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)

        try:
            if type(images[i]) != Cutout2D:
                plt.imshow(images[i].reshape((h, w)))
            else: 
                plt.title(f'{images[i].center_original}')
                plt.imshow(images[i].data.reshape((h, w)))
        except IndexError:
            pass
            

        plt.xticks(())
        plt.yticks(())

def linecut_plot_gallery(images, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)

        try:
            x, row = GetNthRow(images[i], h//2)
            plt.plot(x, row)
        except IndexError:
            pass

        plt.xticks(())
        plt.yticks(())


def compare_results(images, h, w, n_row=3, n_col=4):
    for set in images:
        plot_gallery(set, h, w, n_row, n_col)
