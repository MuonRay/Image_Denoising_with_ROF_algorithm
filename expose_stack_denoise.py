# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 02:27:11 2021

@author: cosmi
"""

import os, numpy
from PIL import Image
import rof
from matplotlib import pyplot as plt  # For image viewing


files   = os.listdir(os.getcwd())
images  = [name for name in files if name[-4:] in [".jpg", ".JPG"]]
width, height = Image.open(images[0]).size


stack   = numpy.zeros((height, width, 3), numpy.float)
counter = 1

#denoise source images

for image in images:
    
    outfile = image[:-3] + "png"
    print( "new filename : " + outfile)
    
    print ("Denoising image " + str(counter))
    im = numpy.array(Image.open(image).convert('L'))
    U,T = rof.denoise(im,im)
    
    fig, ax = plt.subplots()

    image = ax.imshow(U, cmap='gray')
    plt.axis('off')
    #Lock or Unlock Key Bar Here for Mapping/Sampling/Showcasing:
    #create_colorbar(fig, image)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #imageio.imsave(outfile, ndvi)
    fig.savefig(outfile, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)
    
    counter  += 1



#stacking denoised images

imlist  = [name for name in files if name[-4:] in [".png", ".PNG"]]
width2, height2 = Image.open(imlist[0]).size

stack   = numpy.zeros((height2, width2, 3), numpy.float)
counter = 1

for pngimage in imlist:
    print ("Processing image " + str(counter))
    
    image_new = numpy.array(Image.open(pngimage), dtype = numpy.float)
    stack     = numpy.maximum(stack, image_new)
    counter  += 1

stack = numpy.array(numpy.round(stack), dtype = numpy.uint8)

output = Image.fromarray(stack, mode = "RGB")

output.save("exposure_stacked_denoised_image.jpg", "JPEG")



#convert from greyscale to colour

#color_img = cv2.cvtColor(output, cv.CV_GRAY2RGB)
