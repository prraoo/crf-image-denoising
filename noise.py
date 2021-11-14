"""
source: https://bit.ly/2XGzjVk
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""

import numpy as np
import os
import cv2
import glob
import natsort


def noisy(noise_typ,image):

	if noise_typ == "gauss":
		row,col,ch= image.shape
		mean = 0
		var = 100
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		return noisy
	elif noise_typ == "s&p":
		row,col,ch = image.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
		out[coords] = 0
		return out

	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy

	elif noise_typ =="speckle":
		row,col,ch = image.shape
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)
		noisy = image + image * gauss
		return noisy

if __name__ == '__main__':
	input_dir = '//winfs-inf/HPS/prao2/nobackup/uni/u-net/data/cifar10/train/airplane/'
	out_dir_gauss = '//winfs-inf/HPS/prao2/nobackup/uni/u-net/data/cifar10/train/airplane_gauss_noise/'

	if not os.path.exists(out_dir_gauss):
		os.makedirs(out_dir_gauss)

	nimgs = 0
	for img in natsort.natsorted(glob.glob1(input_dir, '*.png')):
		nimgs += 1
		out_img = noisy('gauss', cv2.imread(input_dir + img))
		cv2.imwrite(out_dir_gauss + img, out_img)
		# if nimgs >500:
		# 	break