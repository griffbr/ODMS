# File: eval_odms.py

import IPython, os, glob, numpy as np, odms

def initialize_test_data(eval_set, eval_name):
	test_dir = os.path.join("../", "data", eval_set, eval_name)
	example_list = next(os.walk(test_dir))[1]
	n_examples = len(example_list)

	# Build file lists.
	files = []
	for ex in example_list:
		files.append(sorted(glob.glob(os.path.join(test_dir, ex, "*.png"))))

	# Build lists for camera movement distances and depth.
	dists = []
	if not eval_name == "driving":
		for i in range(n_examples):
			dists.append([float(f.split("/")[-1].split(".png")[0]) / 10E5 for
															f in files[i]])
		depths = dists
	else:
		# Driving data has moving objects, so depth varies from move distance.
		depths = []
		for i in range(n_examples):
			txt =os.path.join("/".join(files[i][0].split("/")[:-1]),"info.txt")
			dist, depth = read_driving_distance(txt, len(files[i]))
			dists.append(dist)
			depths.append(depth)
		# Driving data is stored in reverse order.
		files = [f[::-1] for f in files]
		dists = [d[::-1] for d in dists]
		depths = [d[::-1] for d in depths]

	# Add camera movement distance permutations to robot test set.
	if eval_name == "robot":
		obs_limit = 10; n_obs=len(files[i]); idx=range(n_obs); idxs=[]
		# Vary camera movement range for robot input data.
		for move_idx in [10, 15, 20, 25, 30]:
			n_idx = n_obs - move_idx + 1
			for i in range(n_idx):
				idxs.append(np.linspace(idx[i], idx[i+move_idx-1], obs_limit,
																	dtype=int))
		f_out=[]; d_out=[]
		for i in range(n_examples):
			files_rob = np.array(files[i])
			dists_rob = np.array(dists[i])
			for idx_rob in idxs:
				f_out.append(files_rob[idx_rob])
				d_out.append(dists_rob[idx_rob])
		files = f_out; dists = d_out; depths = d_out

	# Make sure camera movement distance is relative to the starting point.
	dists = [d - d[0] for d in np.array(dists)]

	return {"files": files, "camera_movement": dists, "depths": depths, 
													"n_examples": len(files)}

def read_driving_distance(data_file, n_obs):
	camera_move = np.zeros(n_obs)
	depth = np.zeros(n_obs)
	for i in range(n_obs):
		data_line = odms.util.read_data_line(data_file, i+1)
		camera_move[i] = float(data_line[0])
		depth[i] = float(data_line[1])
	return camera_move, depth
