"""
Demonstration of how to evaluate ODN network on ODMS.
"""

import sys, os, cv2, IPython, torch, numpy as np, _pickle as pickle
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir); sys.path.insert(0,"../")
import odms, odn

# Select trained ODN network and corresponding configuration file.
net_name = "ODNlr_demo"
odn_config = "../config/odnlr.yaml"

# Initiate data loader and odn model.
net, device, m_params = odn.load_model(odn_config)
seg2net = odn.SegmentationMasksToNetwork(m_params)
model_dir = os.path.join("..", "results", "model", net_name)
model_list = sorted([pt for pt in os.listdir(model_dir) if pt.endswith(".pt")])
n_models = len(model_list)

# Select dataset to evaluate.
eval_set = "val" # or "test" once model training and development are complete.
set_list = ["robot", "driving", "normal", "perturb"]
# If running test eval, manually set the model_list based on val performance.
if eval_set == "test" and n_models > len(set_list):
	print("\nWarning!!! Can use only one model weight for each test set.\n")
	IPython.embed()

# Run evaluation.
percent_error=[[] for i in set_list]; abs_error=[[] for i in set_list]
depths=[[] for i in set_list]; model_name=[[] for i in set_list]
with torch.no_grad(): 
	for j, test_set in enumerate(set_list):

		# Get a list of parameters determining how test is going to go.
		test_data = odms.eval.initialize_test_data(eval_set, test_set)
		n=test_data["n_examples"]; ground_truth=np.zeros(n)
		seg2net.set_batch(n)

		print("\nLoading and processing %s (%s examples)." % (test_set, n))
		all_masks=[]; all_moves=[]
		for i, mask_list in enumerate(test_data["files"]):
			ground_truth[i] = test_data["depths"][i][0]
			seg_mask = np.array([cv2.imread(f,cv2.IMREAD_GRAYSCALE)/255 
					for f in mask_list])
			camera_movement = np.array(test_data["camera_movement"][i])
			if seg2net.dim[0] < len(seg_mask):
				input_idx = np.linspace(0, len(seg_mask)-1, seg2net.dim[0], 
						dtype=int)
				seg_mask = seg_mask[input_idx]
				camera_movement = camera_movement[input_idx]
			all_masks.append(seg_mask)
			all_moves.append(camera_movement)

		# Run ODN with correct pre- and post-processing for configuration.
		seg2net.input_data_to_network(np.array(all_masks), np.array(all_moves))
		inputs = seg2net.inputs.to(device)
		inputs_2 = seg2net.inputs_2.to(device)

		print("Processing %s results for %i models." % (test_set, n_models))
		min_error=10e6
		for i, model in enumerate(model_list):
			net = odn.load_weights(net, os.path.join(model_dir, model))
			outputs = net(inputs, inputs_2).cpu().numpy()
			depth = seg2net.network_output_to_depth(outputs[:,0])
			error = np.mean(abs(depth - ground_truth) / ground_truth)
			print("[%s] Mean Percent Error: %.4f" % (model, error))

			# Save best results.
			if error < min_error:
				print("New minimum error!"); min_error=error
				percent_error[j]=error; depths[j]=depth; model_name[j]=model
				abs_error[j] = np.mean(abs(depth - ground_truth))

# Print out final results.
print("\nResults summary for ODMS %s sets." % eval_set)
for i, test_set in enumerate(set_list):
	print("\n%s-%s:" % (test_set, eval_set))
	print("Model: %s" % model_name[i]) 
	print("Mean Percent  Error: %.4f" % percent_error[i]) 
	print("Mean Absolute Error: %.4f (m)" % abs_error[i]) 

# Generate final results file.
result_data = {"Result Name": net_name, "Set List": set_list, "Eval": eval_set, 
				"Percent Error": percent_error, "Absolute Error": abs_error, 
				"Depth Estimates": depths, "Model Names": model_name}
pickle.dump(result_data,open("../results/%s_%s.pk" %(net_name, eval_set),"wb"))