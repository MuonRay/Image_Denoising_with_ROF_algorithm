# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:37:45 2021

@author: cosmi
"""


"""
Experimental UV Absorption Index program using UV-Pass Filter on DJI Mavic 2 Pro 
JPEG 16-bit combo images taken using Ultraviolet-Only Pass Filter 
Useful for Batch Processing Multiple Images

%(c)-J. Campbell MuonRay Enterprises 2020 
This Python script was created using the Spyder Editor
"""
from scipy import misc

import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import pylab
import rof

import imageio
import numpy as np
from matplotlib import pyplot as plt  # For image viewing

#!/usr/bin/python
import getopt
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

#dng reading requires libraw to work

# Open an image
image = misc.imread('DJI_0212.JPG')

# Get the red band from the rgb image, and open it as a numpy matrix
#NIR = image[:, :, 0]
         
#ir = np.asarray(NIR, float)


ir = (image[:,:,0]).astype('float')


# Get one of the IR image bands (all bands should be same)
#blue = image[:, :, 2]

#r = np.asarray(blue, float)

r = (image[:,:,2]).astype('float')


g = (image[:,:,1]).astype('float')


ir = (image[:,:,0]).astype('float')

denoised_g_channel = g


""" negating noise in the output green channel using
  an implementation of the Rudin-Osher-Fatemi (ROF) denoising model
    using the numerical procedure presented in eq (11) A. Chambolle (2005).
     
The ROF model has the interesting property that it finds a smoother version 
of the image while preserving edges and structures.


    Input: noisy input image (grayscale), initial guess for U, weight of
    the TV-regularizing term, steplength, tolerance for stop criterion.

    Output: denoised and detextured image, texture residual. """

U,T = rof.denoise(denoised_g_channel,denoised_g_channel)

pylab.figure()
pylab.gray()
pylab.imshow(U)
pylab.axis('equal')
pylab.axis('off')
pylab.show()


#(NIR + Green)
irg = np.add(ir, U)
       
       
L=0.5;
       
rplusb = np.add(ir, r)
rplusbplusg = np.add(ir, r, U)
rminusb = np.subtract(ir, r)
oneplusL = np.add(1, L)
# Create a numpy matrix of zeros to hold the calculated UVRI values for each pixel
uvri = np.zeros(r.size)  # The UVRI image will be the same size as the input image

# Calculate UV Reflectance Index

uvri = np.true_divide(np.subtract(U, rminusb), np.add(U, rplusb))


# Display the results
output_name = 'UVReflectanceIndex4.jpg'

#a nice selection of grayscale colour palettes
cols1 = ['blue', 'green', 'yellow', 'red']
cols2 =  ['gray', 'gray', 'red', 'yellow', 'green']
cols3 = ['gray', 'blue', 'green', 'yellow', 'red']

cols4 = ['black', 'gray', 'blue', 'green', 'yellow', 'red']

def create_colormap(args):
    return LinearSegmentedColormap.from_list(name='custom1', colors=cols4)

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
        cbar.set_label("UVRI", fontsize=10, x=0.5, y=0.5, labelpad=-25)

fig, ax = plt.subplots()
image = ax.imshow(uvri, cmap=create_colormap(colors))
plt.axis('off')

create_colorbar(fig, image)

extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(output_name, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)
        # plt.show()