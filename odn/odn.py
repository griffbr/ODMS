import torch, IPython
import torch.nn as nn
import torch.nn.functional as F

def pass_arg_for_model(model_name, **kwargs):
	if "ODN" in model_name:
		model = ObjectDepthNetwork(CustomBlock, **kwargs)
	return model

class ObjectDepthNetwork(nn.Module):
	def __init__(self, block, layr=[1], kern=[1], fcs=[512], n1=5, n2=5, 
			n2_in=False, n2_fc=True, avg_pool=7, par_net=False, par_n=1,
			batch_nrm=True, n2_mid=False, mid_rep=0, strd=[1], conv_kern=14,
			max_pool=False, fc_drop=0):
		super(ObjectDepthNetwork, self).__init__()

		# Initial config.
		self.y_in=n2_in; self.y_fc=n2_fc; self.par_net=par_net
		self.n1=n1; self.par_n=par_n; self.inplanes=kern[0]
		self.bn = batch_nrm; self.n2_mid = n2_mid; self.mid_rep = mid_rep
		self.n_fc = len(fcs); self.n_blck = len(kern); self.max_pool = max_pool

		# Parallel process input masks (notes pg. 27 190923).
		if self.par_net: 
			n_conv0 = n1 - par_n + 1 + n2 * n2_in / par_n
			template_idx = range(self.par_n, self.n1)
			idxs = []
			for i in range(par_n):
				if n2_in: idx = [i, i + n1] + template_idx
				else: idx = [i] + template_idx
				idxs.append(idx)
			self.net1_idxs = idxs
		else: n_conv0 = n1 + n2 * n2_in

		# Initial convolution and max pooling.
		self.conv1 = nn.Conv2d(n_conv0, self.inplanes, kernel_size = conv_kern, 
				stride=2, padding=3, bias=False)
		if self.bn: self.bn1 = nn.BatchNorm2d(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(avg_pool, stride=1)
		if self.max_pool:
			self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		# Residual layers.
		self.layer1 = self._make_layer(block,kern[1],layr[0],stride=strd[0])
		if self.par_net: self.inplanes *= par_n
		if self.n2_mid: self.inplanes += n2 * mid_rep
		self.layer2 = self._make_layer(block,kern[2],layr[1],stride=strd[1])
		self.layer3 = self._make_layer(block,kern[3],layr[2],stride=strd[2])
		if self.n_blck > 4:
			self.layer4 = self._make_layer(block,kern[4],layr[3],stride=strd[3])

		# Fully connected layers.
		self.fc0 = nn.Linear(fcs[0] + n2 * n2_fc, fcs[1]) # If adding distance.
		#self.fc0 = nn.Linear(fcs[0], fcs[1])
		if self.n_fc > 2: self.fc1 = nn.Linear(fcs[1] + n2 * n2_fc, fcs[2])
		if self.n_fc > 3: self.fc2 = nn.Linear(fcs[2] + n2 * n2_fc, fcs[3])
		self.fc = nn.Linear(fcs[-1] * block.expansion + n2 * n2_fc, 1)
		if fc_drop > 0: 
			self.drop = True; self.dropout = nn.Dropout(p=fc_drop)
		else: self.drop = False

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride !=1 or self.inplanes != planes * block.expansion:
			if self.bn:
				downsample = nn.Sequential(
						nn.Conv2d(self.inplanes, planes * block.expansion,
								kernel_size=1, stride=stride, bias=False),
						nn.BatchNorm2d(planes * block.expansion),
				)
			else:
				downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
								kernel_size=1, stride=stride, bias=False)
		layers = []
		layers.append(block(self.inplanes,planes,stride,downsample,bn=self.bn))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes,bn=self.bn))
		return nn.Sequential(*layers)

	def forward(self, x, y):
		# Concatenate with Residual Optics (see notes pg. 16 on 190710)
		if self.y_in:
			# y also includes camera move distance.
			x = torch.cat((x, y), 1) # Add camera move distance.
		if self.par_net:
			# Process each mask independently and then cat.
			x0 = self.net_1(x[:, self.net1_idxs[0]]) 
			for i in range(1, self.par_n):
				x0 = torch.cat((x0, self.net_1(x[:,self.net1_idxs[i]])), 1)
			x = x0
		else:
			x = self.conv1(x) # 480x640 (224x224) [56x56, 480x480]
			if self.bn: x = self.bn1(x)
			x = self.relu(x)
			if self.max_pool:
				x = self.maxpool(x) # 240x320 (112x112) [28x28, 240x240]
			x = self.layer1(x) # 120x160 (56x56) [14x14, 120x120]

		# Potential distance / residual concatenate point.
		if self.n2_mid:
			y0 = y
			for i in range(1, self.mid_rep):
				y0 = torch.cat((y0, y), 1)
			x = torch.cat((x, y0), 1)

		x = self.layer2(x) # 60x80 (28x28) [7x7, 60x60]
		#print('pre layer 3'); IPython.embed()
		x = self.layer3(x) # 30x40 (14x14) [4x4, 30x30]
		if not hasattr(self, 'n_blck'):
			self.n_blck = 5 # remove for later models
		#print('pre layer 4'); IPython.embed()
		if self.n_blck > 4:
			x = self.layer4(x) # 15x20 (7x7) [2x2, 15x15]
		#x = self.layer5(x) # 8x10 (4x4) [1x1, 8x8]
		#print('pre average pool'); IPython.embed()

		if not hasattr(self, 'drop'):
			self.drop = False # remove for later models

		x = self.avgpool(x) # 1x1
		#print('post average pool'); IPython.embed()
		x = x.view(x.size(0), -1)
		if self.y_fc:
			# Add camera move distance to fc layers.
			distances = y[:,:,0,0]
			x = torch.cat((x, distances), 1) 
		x = self.fc0(x)
		x = self.relu(x)
		if self.drop: x = self.dropout(x)
		if self.n_fc >2:
			if self.y_fc: x = torch.cat((x, distances), 1) 
			x = self.fc1(x)
			x = self.relu(x)
			if self.drop: x = self.dropout(x)
		if self.n_fc > 3:
			if self.y_fc: x = torch.cat((x, distances), 1) 
			x = self.fc2(x)
			x = self.relu(x) # Uncommented and moved into if statement 191101.
			if self.drop: x = self.dropout(x)
		if self.y_fc: x = torch.cat((x, distances), 1) 
		x = self.fc(x)
		
		return x

	def net_1(self, x):
		x = self.conv1(x) # 480x640 (224x224) [56x56, 480x480]
		if self.bn: x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x) # 240x320 (112x112) [28x28, 240x240]
		return x
		
class CustomBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride=1, downsample=None, bn=True):
		super(CustomBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		if bn: self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		if bn: self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.bn = bn
		#self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		if self.bn: out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		if self.bn: out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	# 3x3 convolution with padding
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
			padding=dilation, groups=groups, bias=False, dilation=dilation)