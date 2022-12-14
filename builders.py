''' Builders functions that create variables and files '''

import numpy as np
import matplotlib.pyplot as plt

'''     
Usefull functions for images :
- image   -> tensor : img = plt.imread("image.png")
- tensor -> display : plt.imshow(img)
- tensor -> image   : plt.imsave("image2.png",img)

Formats:
- Pyplot: tensors (N,N,3) with n=N^2 the number of pixel.
- Our functions: matrices (N^2,3)=(n,3), vectorization of the tensor.
'''


def image2array(name):
    ''' Read an image file and put it on a array
    Input
            name: string, name of an image file with extension
    Output
            :array of shape (N^2,3), the array associated to the (N,N)-image '''
    return np.reshape(plt.imread(name), (-1,3)) / 256 # à garder?

def array2matrix(img):
    ''' Write an image file from an array
    Input
            img: array of shape (N^2,3), the array associated to the (N,N)-image
    Output
            array of shape (N^2,3), the array associated to the (N,N)-image '''
    N = int(round(np.sqrt(np.shape(img)[0])))
    return np.reshape(img,(N,N,3))
    return

def array2image(name, img):
    ''' Write an image file from an array
    Input
            name: string, the future name of an image file with extension
            img: array of shape (N^2,3), the array associated to the (N,N)-image
    No Output '''
    plt.imsave(name, array2matrix(img),vmin=0, vmax=1) # entre 0 et 1
    return

def array2cost(source_img,target_img):
    '''Compute the cost matrix associated to two arrays representing images
    Input
            source_img: img: array of shape (N^2,3), the array associated to the source (N,N)-image
            target_img: img: array of shape (N^2,3), the array associated to the target (N,N)-image
    Output
            C: array of shape (n,n), the cost matrix associated to the two images
            m1, m2: array of shape(n,1), uniform stochastic vectors'''
    source_img = source_img[:,np.newaxis, :]
    target_img = target_img[np.newaxis,: , :]
    C = np.sum((source_img - target_img)**2, axis=2)
    n = np.shape(C)[0]
    m1, m2 = np.ones((n,1))/n, np.ones((n,1))/n
    return C,m1,m2

def image2cost(source_name,target_name):
    '''Compute the cost matrix associated to two images files
    Input
            source_name: string, name of the source image file with extension
            target_name: string, name of the target image file with extension
    Output
            C: array of shape (n,n), the cost matrix associated to the two images
            m1, m2: array of shape(n,1), uniform stochastic vectors'''
    source_img = image2array(source_name)
    target_img = image2array(target_name)
    return array2cost(source_img,target_img)

def transfer_color(P,img):
    ''' transfert the color from a source image according to coupling
    Input: 
            P: array of shape (n,n), coupling
            img: array of shape (n,3), the array associated to an image
    Output
            : array of shape (n,3), the array associated to the new colored image '''
    q =  np.sum(P,axis=1)
    return P.T@img / q[:,np.newaxis] # ATTENTION erreur dans la desciption du projet!