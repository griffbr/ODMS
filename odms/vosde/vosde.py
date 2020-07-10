"""
VOS-DE baseline from:
Video Object Segmentation-based Visual Servo Control and Object Depth
Estimation on a Mobile Robot
Brent Griffin, Victoria Florence, and Jason J. Corso
IEEE Winter Conference on Applications of Computer Vision (WACV), 2020
"""

import os, IPython, numpy as np
from skimage.measure import label

def estimate_depth(masks, camera_movement, n_input=10):
	# Initialize inputs.
	n_observations = len(masks)
	if n_input < n_observations:
		input_idx = np.linspace(0, n_observations-1, n_input, dtype=int)
		masks = [masks[i] for i in input_idx]
		camera_movement = camera_movement[input_idx]
	masks_single = [largest_region_only(mask) for mask in masks]
	area_sqrt = [mask.sum()**0.5 for mask in masks_single]

	# Find least squares solution for depth estimate (see paper for details).
	A = np.zeros(shape=(n_input, 2))
	b = np.zeros(shape=(n_input, 1))
	A[:,0] = area_sqrt
	A[:,1] = 1
	b = camera_movement * area_sqrt
	try:
		x = np.matmul( np.matmul( np.linalg.inv( np.matmul(A.T,A)), A.T), b)
	except:
		print("Matrix A.T A is not invertable! Depth is not solved by VOS-DE.")
		x = np.zeros(2)
	return -x[0]

def largest_region_only(init_mask):
	labels = label(init_mask)
	bin_count = np.bincount(labels.flat)
	if len(bin_count) > 1:
		mask_bin = np.argmax(bin_count[1:]) + 1
		single_mask = labels == mask_bin
	else:
		single_mask = init_mask
	return single_mask
