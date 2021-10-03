# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:37:45 2021

@author: cosmi
"""

"""
Denoised Vegetation Index Mapping program using DJI Mavic 2 Pro
JPEG 16-bit combo images taken using InfraBlue Filter
%(c)-J. Campbell MuonRay Enterprises 2021
% creative commons For non-profit use only
This Python script was created using the Spyder Editor
"""

import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import pylab
import rof

from scipy import misc
import imageio
import numpy as np
from matplotlib import pyplot as plt  # For image viewing

#!/usr/bin/python
import os
import getopt
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap



#a nice selection of grayscale colour palettes
cols1 = ['blue', 'green', 'yellow', 'red']
cols2 =  ['gray', 'gray', 'red', 'yellow', 'green']
cols3 = ['gray', 'blue', 'green', 'yellow', 'red']

cols4 = ['black', 'gray', 'blue', 'green', 'yellow', 'red']

def create_colormap(args):
    return LinearSegmentedColormap.from_list(name='custom1', colors=cols3)

#colour bar to match grayscale units
def create_colorbar(fig, image):
        position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
        norm = colors.Normalize(vmin=-1., vmax=1.)
        cbar = plt.colorbar(image,
                            cax=position,
                            orientation='horizontal',
                            norm=norm)
        cbar.ax.tick_params(labelsize=6)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label("NDVI", fontsize=10, x=0.5, y=0.5, labelpad=-25)



for infile in os.listdir("./"):
    print( "file : " + infile)
    if infile[-3:] == "jpg" or infile[-3:] == "JPG" :
       # print "is tif or DNG (RAW)"
       outfile = infile[:-3] + "jpg"
       rgb = misc.imread(infile)
       
       
       print( "new filename : " + outfile)
       # Extract Red, Green and Blue channels and save as separate files
       

       R = rgb[:,:,0]
       G = rgb[:,:,1]
       B = rgb[:,:,2]
       
              # Get the red band from the rgb image, and open it as a numpy matrix
#NIR = image[:, :, 0]
#ir = np.asarray(NIR, float)
              
       ir = (R).astype('float')
       
# Get one of the IR image bands (all bands should be same)
#blue = image[:, :, 2]

#r = np.asarray(blue, float)
       
       r = (B).astype('float')
       
       #denoise
       
       denoised_ir_channel = ir
       
       
       U,T = rof.denoise(denoised_ir_channel,denoised_ir_channel)
       
       #pylab.figure()
       #pylab.gray()
       #pylab.imshow(U)
       #pylab.axis('equal')
       #pylab.axis('off')
       #pylab.show()


# Create a numpy matrix of zeros to hold the calculated NDVI values for each pixel
  # The NDVI image will be the same size as the input image

       
       ndvi = np.zeros(r.size)       
       
# Calculate NDVI
       
       
       ndvi = np.true_divide(np.subtract(U, r), np.add(U, r))
       fig, ax = plt.subplots()

       image = ax.imshow(ndvi, cmap=create_colormap(colors))
       plt.axis('off')
       #Lock or Unlock Key Bar Here for Mapping/Sampling/Showcasing:
       #create_colorbar(fig, image)
       extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
       #imageio.imsave(outfile, ndvi)
       fig.savefig(outfile, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)

        # plt.show()
       
       
#       rgb = raw.postprocess()

        # plt.show()