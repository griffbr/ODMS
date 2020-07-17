# File: generate_random_mask.py

import IPython, os, numpy as np, scipy.ndimage
from .bezier_contour import get_random_points, get_bezier_curve

def generate_masks(rad, rot, n_pt, size, scale, im_dim, n_obs, pert, std_dev):
	masks = np.zeros(shape=(n_obs, im_dim[0], im_dim[1]), dtype="float32")

	# Generate mask contours using Bezier splines.
	a = get_random_points(n=n_pt, scale=size)
	x, y, _ = get_bezier_curve(a, rad=rad, edgy=rot)
	for i in range(n_obs):
		masks[i]=scaled_mask_from_points(x,y,size,scale[i],im_dim[0],im_dim[1])

	# Add perturbations.
	if pert:
		for i in range(n_obs):
			masks[i] = random_erosion_dilation(masks[i], std_dev)
	return masks

def scaled_mask_from_points(x, y, pt_scale, scale, h, w):
	image = np.zeros(shape=(h,w))
	x_scale = x * scale
	y_scale = y * scale
	off = np.array([h/2, w/2], dtype=int) - int(scale * pt_scale / 2)
	x_idx = x_scale.astype(int) + off[0]
	y_idx = y_scale.astype(int) + off[1]
	# Apply out of image threshold.
	if max(x_idx) > (h - 1):
		x_idx[x_idx > h - 1] = h - 1
	if min(x_idx) < 0:
		x_idx[x_idx < 0] = 0
	if max(y_idx) > (w - 1):
		y_idx[y_idx > w - 1] = w - 1
	if min(y_idx) < 0:
		y_idx[y_idx < 0] = 0
	image[x_idx, y_idx] = 1
	mask = scipy.ndimage.morphology.binary_fill_holes(image)
	return mask

def random_erosion_dilation(mask, std_dev):
	# Add random perturbation. See Equation (20) in paper for full details.
	kernel_size = int(np.round(np.abs(np.random.normal(scale=std_dev))))
	if kernel_size > 0:
		if np.random.randint(0, 2):
			mask = scipy.ndimage.morphology.binary_dilation(mask, 
														iterations=kernel_size)
		else:
			mask = scipy.ndimage.morphology.binary_erosion(mask, 
														iterations=kernel_size)
	return mask
