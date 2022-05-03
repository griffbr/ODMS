import torch, os, IPython, numpy as np, yaml, cv2
from copy import deepcopy
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from .odn import *

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(model_config):

	# Parameter initialization.
	m_param = yaml.full_load(open(model_config))
	m_param["network"] = {}
	m_param["network"]["layr"] = m_param["layer"]
	m_param["network"]["kern"] = m_param["kernal"]
	m_param["network"]["strd"] = m_param["stride"]
	m_param["network"]["fc_drop"] = m_param["fc_dropout"]
	m_param["network"]["fcs"] = [m_param["kernal"][-1], m_param["kernal"][-1]]
	m_param["network"]["n1"] = m_param["dim"][0] + len(m_param["template"])
	m_param["network"]["avg_pool"] = int(round(m_param["dim"][1] / 2.**5))
	m_param["network"]["n2"] = m_param["dim"][0] - \
			(1 + m_param["normalized_distance_input"])

	# Take model configuration parameters and set up network architecture.
	net = pass_arg_for_model(m_param["model_name"], **m_param["network"])

	# Randomly select between both GPUs (distribute training).
	gpus = ["cuda:0", "cuda:1"]
	idx = np.random.randint(0,2)
	try:
		device = torch.device(gpus[idx])
		net.to(device)
	except:
		try:
			device = torch.device(gpus[idx-1])
			net.to(device)
		except:
			device = torch.device("cpu")
			net.to(device)

	return net, device, m_param

def load_training_params(train_config):
	train = yaml.full_load(open(train_config))
	train["display_iter"] = int(train["train_iter"] / train["n_display"])
	train["save_iter"] = np.linspace(0, train["train_iter"], 
			train["n_train_model_saves"] + 1).astype(int)[1:]
	return train

def load_weights(net, weight_file):
	# Load weights onto already initialized network model.
	try:
		net.load_state_dict(torch.load(weight_file))
	except:
		net.load_state_dict(torch.load(weight_file, map_location=lambda 
				storage, loc: storage))
	return net

def resize_masks(args):
	mask_set=args[0]; dim=args[1]
	masks_resized = np.zeros(shape=(dim[0], dim[1], dim[2]), dtype="float32")
	for i, mask in enumerate(mask_set):
		masks_resized[i] = cv2.resize(mask.astype('float32'), (dim[1], dim[2]),
				interpolation=cv2.INTER_AREA)
	return masks_resized

class SegmentationMasksToNetwork:
	"""
	Converts segmentation masks and camera distances to network input.
	"""

	def __init__(self, params, n_batch=1):
		"""
		Args:
			params (dict): Input/output parameters for ODN network.
			n_obs (int): Number of bounding box and movement inputs to network.
			n_bat (int): Number of input sets batched together.
		"""
	
		# Misc. Initizlization.
		self.dim = params["dim"]
		self.extra_in = params["network"]["n2"] 
		self.n_template = len(params["template"])
		self.set_batch(n_batch)
		self.template = params["template"] 
		self.load_templates(params["template_path"])
		self.norm_dist = params["normalized_distance_input"]
		self.scale_out = params["scale_out"]
		self.scale_n = params["scale_n"]
		self.n_cores = cpu_count()

	def set_batch(self, n_batch):
		# Change network input batch size.
		self.batch = n_batch
		self.inputs = torch.zeros(n_batch, self.dim[0] + self.n_template,
				self.dim[1], self.dim[2], dtype=torch.float)
		self.inputs_2 = torch.zeros(n_batch, self.extra_in, self.dim[1] // 2**3, 
				self.dim[2] // 2**3, dtype=torch.float)
		self.labels = torch.zeros(n_batch, 1, dtype=torch.float)
		self.move_ranges = np.zeros(self.batch)

	def load_templates(self, template_dir="./set_input/"):
		print("Loading templates.")
		for i, template_img in enumerate(self.template):
			img = cv2.imread('%s%s.png' % (template_dir, template_img), 
				cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (self.dim[1], self.dim[2]))
			# img = 2 * (img / 255.0) - 1 # Scale between [-1,1]
			img = img / 255.0 # Scale between [0,1]
			self.inputs[:,self.dim[0]+i] = torch.from_numpy(img)

	def input_data_to_network(self, seg_masks, camera_movements, depths=[]):
		# Convert segmentation and camera data to network input and labels.

		# Make sure the number of inputs are correct for the network.
		n_observations = len(seg_masks[0])
		if self.dim[0] < n_observations:
			input_idx = np.linspace(0, n_observations-1, self.dim[0], dtype=int)
			seg_masks = seg_masks[:, input_idx]
			camera_movements = camera_movements[:, input_idx]

		# Resize masks for network input (parallelize for faster batch resize).
		par_args = [(mask, self.dim) for mask in seg_masks]
		pool = ThreadPool(self.n_cores)
		par_masks = pool.map(resize_masks, par_args)
		pool.close(); pool.join()
		self.inputs[:,:self.dim[0]] = torch.from_numpy(np.array(par_masks))

		"""# Resize masks for network input.
		for i in range(self.batch):
			for j in range(self.dim[0]):
				self.inputs[i,j] = torch.from_numpy(cv2.resize(
						seg_masks[i,j].astype('float32'), 
						(self.dim[1], self.dim[2])))"""

		# Prepare camera movement inputs.
		for i in range(self.batch):
			move_relative = camera_movements[i]-camera_movements[i,0]
			if self.norm_dist: # Normalized input/output depth prediction.
				self.move_ranges[i] = deepcopy(move_relative[-1])
				move_relative = move_relative[1:-1] / self.move_ranges[i]
			else: # Predict metric object depth.
				move_relative = move_relative[1:]
			move_relative = torch.from_numpy(move_relative)
			for j, move in enumerate(move_relative): self.inputs_2[i,j] = move

		# Make ground truth labels (scale, metric depth, or normalized depth).
		if depths != []:
			if self.scale_out: 
				# Scale change is proportional to depth (see Eq. 13-14 in paper).
				l1 = depths + self.move_ranges
				if self.scale_n: labels = depths / l1
				else: labels = l1 / depths
			else:
				labels = deepcopy(depths) # Metric depth.
				if self.norm_dist: # Normalized depth.
					labels /= self.move_ranges 
			self.labels[:,0] = torch.from_numpy(labels)

	def network_output_to_depth(self, outputs):
		# Convert network outputs to depth.

		# Convert from change in scale to change in depth.	
		if self.scale_out:
			if self.scale_n:
				depths = self.move_ranges * outputs / (1 - outputs)
			else: 
				depths = self.move_ranges / (outputs - 1)
		else:
			depths = outputs # Direct metric depth prediction. Or...
			if self.norm_dist: # Scale normalized depth using camera movement.
				depths *= self.move_ranges

		return depths
