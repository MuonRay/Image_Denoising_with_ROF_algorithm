# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 00:56:27 2021

@author: cosmi
"""
import numpy as np


def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
    
    
    
  """ 
  Removing noise from images is important for many applications, from making 
your personal photos look better to improving the quality of satellite and increasingly
drone images.

In image processing denoising functionally looks like we are smoothing out the image.
but just what is it we are smoothing out to remove the noise? 

The ROF model has the interesting property that it finds a smoother version 
of the image while preserving edges and structures.

The underlying mathematics of the ROF model and the solution techniques are 
quite advanced and are showcased fully in the paper:

Nonlinear total variation based noise removal algorithms*
Leonid I. Rudin 1, Stanley Osher and Emad Fatemi 
Cognitech Inc., 2800, 28th Street, Suite 101, Santa Monica, CA 90405, USA. circa 1992. 

Its interesting to note that the authors base their work on work done previously 
by Geman and Reynolds in which they propse to minimisae the non-linear functionals 
asscoaited with noise in the total varience by use of simulated annealing, a
metaheursitics technique, which would have been very slow to do using the computers 
available in the late 1980s something
which drove the development of the ROF denoising model in the first place.
The authors wanted a fast solver that could find a reasonably good local minima
rather tha the ideal global minima. 

Here I'll give a brief, 
simplified introduction before showing how to implement a ROF solver based 
on an algorithm by Chambolle.

The solver used in this code is a modified version that uses grdient descent/
reprojection method to achieve total variance (TV) minimization/regularization
The integral of the gradient across an image, any image, will produce the total varience
in tha image. now, for noisy images the total varience will be higher. knowing this,
denoising techniques have been developed that essentially minimise the total varience
in the matrix element of an image and then reproject that image onto the original by substracting
the imaginary form of the original image matrix element which contains the residiual texture
of the image. so textures are effectively removed when we want to do TV minimization/regularization.



To minimise the total varience in the matrix element different algorithms can be used
but one of the most popular is gradient descent, which is simular to the simulated annealing
technique originally proposed but more computationally tractable.


in gradient descent the image matrix containing the greyscale pixel values are
essentially represented as an energy surface, whereby 
we want to descend into the global minima. the different values in the matrix represent 
the interaction energies of the nearest neighbour in this 2D energy surface. 
different algorithms can use
different equations to represent the interaction energies, depending on the rate at which
the interaction energies converge to a global minima in a certain steplength of 
iteration of the algorithm. 



Interesting Note: When implented using periodic boundary conditions the TV minimization using 
an iteractive gradient descent on an image performs 2 transformations on that
images rectangular image domain where the greyscale pixel values exist. 2 transformations 
on a rectangle result in the formation of a torus in a transformation which 
preserves the images pixel data but changes the domain topology. 
  
  
  
  
    An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
    using the numerical procedure presented in equation 11 on pg 15 of
    A. Chambolle (2005)
    http://www.cmap.polytechnique.fr/preprint/repository/578.pdf
    
        
        Input:
        im - noisy input image (grayscale)
        U_init - initial guess for U
        tv_weight - weight of the TV-regularizing term
        tau - steplength in the Chambolle algorithm
        tolerance - tolerance for determining the stop criterion
    
        Output:
        U - denoised and detextured image (also the primal variable)
        T - texture residual
        




    Input: noisy input image (grayscale), initial guess for U, weight of
    the TV-regularizing term, steplength, tolerance for stop criterion.

    Output: denoised and detextured image, texture residual. """

  m,n = im.shape # size of noisy image

  # initialize
  U = U_init
  Px = im # x-component to the dual field
  Py = im # y-component of the dual field
  error = 1
  iteration = 0

  

  while (error > tolerance):
    Uold = U

    # gradient of primal variable
    GradUx = np.roll(U,-1,axis=1)-U # x-component of U's gradient
    GradUy = np.roll(U,-1,axis=0)-U # y-component of U's gradient

    # update the dual varible
    PxNew = Px + (tau/tv_weight)*GradUx
    PyNew = Py + (tau/tv_weight)*GradUy
    NormNew = np.maximum(1,np.sqrt(PxNew**2+PyNew**2))

    Px = PxNew/NormNew # update of x-component (dual)
    Py = PyNew/NormNew # update of y-component (dual)
    """
    the function roll(), which, as the name suggests, “rolls” the values of an array 
    cyclically around an axis. This is very convenient for computing neighbor differences, 
    in this case for derivatives. We also used linalg.norm(), which measures the difference
    between two arrays (in this case, the image matrices U and Uold).
    """
    # update the primal variable
    RxPx = np.roll(Px,1,axis=1) # right x-translation of x-component
    RyPy = np.roll(Py,1,axis=0) # right y-translation of y-component

    DivP = (Px-RxPx)+(Py-RyPy) # divergence of the dual field.
    U = im + tv_weight*DivP # update of the primal variable

    # update of error
    error = np.linalg.norm(U-Uold)/np.sqrt(n*m);

    iteration += 1;
    
    #The texture residual
  T = im - U
  #print: 'Number of ROF iterations: ', iteration
    
  return U,T # denoised image and texture residual




