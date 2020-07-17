# File: data_gen.py

import IPython, os, cv2, numpy as np, yaml
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from .generate_random_mask import generate_masks

class DataGenerator:
	"""
	DataGenerator generates random data for ODMS.
	"""

	def __init__(self, config_file):
		self.set_params(config_file)
		self.z1_range = self.z_lim[1] - self.z_lim[0] - self.move_min
		self.n_cores = cpu_count()

	def set_params(self, config_file):
		params = yaml.full_load(open(config_file))
		for _, key in enumerate(params.keys()):
			setattr(self, key, params[key])

	def generate_examples(self, n_ex):
		# Generate n_ex random training examples.
		# See Section 5.1 of our paper for more details.

		# Determine random distances.
		dist = np.zeros(shape=(n_ex, self.n_observations))
		stpts = self.z_lim[0] + np.random.rand(n_ex) * self.z1_range
		zn_range = self.z_lim[1] - stpts
		endpts = stpts + self.move_min + np.random.rand(n_ex) * zn_range
		z_range = endpts - stpts
		for i in range(1, self.n_observations-1):
			dist[:,i] = stpts + np.random.rand(n_ex) * z_range
		dist[:,1:-2].sort(axis=1)
		dist[:,0] = stpts
		dist[:,-1] = endpts

		# Determine camera movement and ground truth object depth.
		camera_movement = np.array([d - d[0] for d in dist])
		depth = dist[:,0]

		# Generate random mask parameters.
		seg_masks = np.zeros(shape=(n_ex, self.n_observations, 
						self.image_dim[0], self.image_dim[1]), dtype="float32")
		rads = select_random(self.bez_radius, n_ex)
		rots = select_random(self.bez_rotate, n_ex)
		n_pts = select_random(self.n_points, n_ex)
		sizes = select_random(self.size, n_ex)
		scales = self.z_lim[0]/dist

		# Parallelize mask generation for faster operation.
		par_args = [(rad, rot, n_pt, size, scale, self.image_dim, 
						self.n_observations, self.perturb, self.std_dev)
						for rad, rot, n_pt, size, scale
						in zip(rads, rots, n_pts, sizes, scales)]
		pool = ThreadPool(self.n_cores)
		par_masks = pool.map(par_wrapper, par_args)
		pool.close()
		pool.join()
		seg_masks[:] = par_masks[:]	

		return seg_masks, camera_movement, depth

def select_random(options, n_ex):
	idx = np.random.randint(0, len(options), n_ex)
	return [options[i] for i in idx]

def par_wrapper(args):
	return generate_masks(*args)
