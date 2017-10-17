# Credit: Li-Hsin Tseng
import argparse
import torch
import os 
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class LastLayer(nn.Module):

	def __init__(self):
		super(LastLayer, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(4096, 200),
		)
		self.criterion = nn.CrossEntropyLoss()

	def forward(self, x):
		x = self.fc(x)
		_, pred_label = torch.max(x.data, 1)
		return pred_label.numpy()[0]

	def train(self, length):
		optimizer = optim.SGD(self.parameters(), lr=0.01)
		cnt, loss_ave = 0, 0
		# for batch_idx, (data, target) in enumerate(self.train_loader):
		for batch_idx in range(3104):
			optimizer.zero_grad()
			# target = torch.autograd.Variable(target)
			if batch_idx % 32 == 0:
				data = torch.load(root+'data/forLastLayer/forLastLayer_'+str(int((batch_idx)/32)+1)+'.pt')
			tmp = batch_idx%32

			x, target = data[tmp]
			x = self.fc(x)
			'''
			_, pred_label = torch.max(x.data, 1)
			res = pred_label.numpy()[0]
			if res == target.data.numpy()[0]: cnt += 1
			'''
			loss = self.criterion(x, target)
			loss_ave += loss
			loss.backward()
			optimizer.step()

		print('loss ave = ' + str(loss_ave/3104))
		'''
		for para in self.parameters():
			print(para.data)
		print('training accuracy:', end = '')
		print(cnt/(3104))
		'''

class AlexNet(nn.Module):
	def __init__(self, data_path, dataBool, num_classes=1000):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)
		self.criterion = nn.CrossEntropyLoss()
		if dataBool:
			train_batch_size = 32
			train_transform = transforms.Compose(
			[transforms.Scale(224), transforms.ToTensor()]) 
			train_data = datasets.ImageFolder(data_path+'train', transform=train_transform)
			self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)

			self.classes = train_data.classes

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x
		# _, pred_label = torch.max(x.data, 1)
		# return pred_label.numpy()[0]

	def train(self, path):
		# optimizer = optim.SGD(self.parameters(), lr=0.01)
		tmp, cnt = [], 0
		for batch_idx, (data, target) in enumerate(self.train_loader):
			print(batch_idx)
			data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
			x = self.features(data)
			x = x.view(x.size(0), 256 * 6 * 6)
			x = self.classifier(x)
			tmp.append((x, target))
			# loss = self.criterion(x, target)
			# loss.backward()
			# optimizer.step()
			if batch_idx%32 == 31:
				torch.save(tmp, path+'_'+str(int(batch_idx/32)+1)+'.pt')
				tmp = []

		#torch.save(tmp, path+'_'+str(int(batch_idx/30+1))+'.pt')

def alexnet(data_path, pretrained=False, **kwargs):
    model = AlexNet(data_path, True)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

if __name__ == "__main__":
	root = os.getcwd() + '/'
	# python3 train.py --data /tiny/imagenet/dir/ --save /dir/to/save/model/
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', metavar='DIR', default='data/tiny-imagenet-200/',
	                    help='/tiny/imagenet/dir/')
	parser.add_argument('--save', metavar='DIR', default='model/my_model.pt',
	                    help='/dir/to/save/model/')

	args = parser.parse_args()
	# one time thing
	model = alexnet(root+args.data, pretrained = True)
	last_layer = LastLayer()

	last_layer = torch.load(root+args.save)
	test_batch_size, epoch_num = 1, 1000
	test_transform = transforms.Compose([transforms.Scale(224), transforms.ToTensor()])
	test_data = datasets.ImageFolder(root=root+args.data+'test', transform=test_transform)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

	# https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766/11
	#for param in model.parameters():
    # 	param.requires_grad = False
	#model.classifier._modules['6'] = nn.Linear(4096, 200)

	# https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/10
	mod = list(model.classifier.children())
	mod.pop()
	# mod.append(torch.nn.Linear(4096, 200))
	new_classifier = torch.nn.Sequential(*mod)
	model.classifier = new_classifier

	### one time thing
	#model.train(root+'data/forLastLayer/forLastLayer')
	last_layer = torch.load(root+args.save)
	for i in range(epoch_num):
		# 224*224
		cnt = 0
		last_layer.train(len(model.train_loader))
		
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
			res = model.forward(data)
			res = last_layer.forward(res)
			if res == target.data.numpy()[0]: cnt += 1
		print('epoch num : ' + str(i))
		print('testing accuracy:', end = '')
		print(cnt/len(test_loader))	
	
		torch.save(last_layer, root+args.save)

		# https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/7
		#last_layer.save_state_dict(root+args.save)

