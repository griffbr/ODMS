"""
Demonstration of how to train ODN network on ODMS.
"""

import sys, os, cv2, IPython, torch
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.insert(0,"../")
import odms, odn

net_name = "ODNlr"

# Select configuration.
datagen_config = "../config/standard_data.yaml" # Other settings in directory.
odn_config = "../config/odnlr.yaml"
train_config = "../config/train_demo.yaml"

# Initiate data generator and model.
odms_data = odms.data_gen.DataGenerator(datagen_config)
net, device, m_params = odn.load_model(odn_config)
train = odn.load_training_params(train_config)
seg2net = odn.SegmentationMasksToNetwork(m_params, train["batch_size"])

# Initiate training!
model_dir = os.path.join("../results", "model", net_name)
os.makedirs(model_dir, exist_ok=True)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0)

running_loss=0.0; ct=0
print("Starting training for %s." % net_name)
while ct < train["train_iter"]:

	# Generate examples for ODMS training (repeat for each training iteration).
	seg_masks, camera_movements, depths = odms_data.generate_examples(
			seg2net.batch)

	# Network inputs and labels, forward pass, loss, and gradient.
	seg2net.input_data_to_network(seg_masks, camera_movements, depths)
	inputs, inputs_2 = seg2net.inputs.to(device), seg2net.inputs_2.to(device)
	labels = seg2net.labels.to(device)
	outputs = net(inputs, inputs_2).to(device)
	loss = criterion(outputs, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	running_loss += loss.item()

	# Print progress details and save model at set interval.
	ct += 1
	if ct % train["display_iter"] == 0:
		cur_loss = running_loss / train["display_iter"]
		print("[%9d] loss: %.6f" % (ct, cur_loss))
		running_loss = 0.0
	if ct in train["save_iter"]:
		torch.save(net.state_dict(), "%s/%s_%09d.pt" % (model_dir,net_name,ct))
		print("[%9d] interval model saved." % ct)
