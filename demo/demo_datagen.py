"""
Demonstration of how to generate new training data on ODMS.
"""

import sys, os, cv2, IPython
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.insert(0,"../")
import odms

datagen_config = "../config/standard_data.yaml" # Other settings in directory.
n_examples = 20 # Configure for batch size if training.
save_examples = False

# Initiate data generator.
odms_data = odms.data_gen.DataGenerator(datagen_config)

# Generate examples for ODMS training (repeat for each training iteration).
seg_masks, camera_movements, depths = odms_data.generate_examples(n_examples)

"""
Use generated data to train your own network to predict depths given seg_masks 
and camera_movements. See paper for ideas on possible initial configurations.
"""

# Save generated examples as a static data set (optional).
if save_examples:
	result_dir = "../data/example_generated_data/"
	for i in range(n_examples):
		example_dir = os.path.join(result_dir, "%.4d" % i)
		os.makedirs(example_dir, exist_ok=True)
		for j, mask in enumerate(seg_masks[i]):
			depth = depths[i] + camera_movements[i,j]
			cv2.imwrite("%s/%.0f.png" % (example_dir, depth*1E6), mask*255)
