# Credit: Li-Hsin Tseng
import argparse
import cv2, time, os
import numpy as np
import torch
import torch.nn as nn
import train
import torch.utils.model_zoo as model_zoo
# python3 test.py --model /dir/containing/model/

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

def cam(model_dir, idx = 0):
		# https://softwarerecs.stackexchange.com/questions/18134/python-library-for-taking-camera-images
		model = train.AlexNet(root+'data/tiny-imagenet-200/', True)
		model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
		
		mod = list(model.classifier.children())
		mod.pop()
		# mod.append(torch.nn.Linear(4096, 200))
		new_classifier = torch.nn.Sequential(*mod)
		model.classifier = new_classifier
		last_layer = LastLayer()
		last_layer = torch.load(model_dir)
		#last_layer.load_state_dict(torch.load(model_dir))

		camera = cv2.VideoCapture(idx)
		time.sleep(0.1) 
		font = cv2.FONT_HERSHEY_SIMPLEX
		#print(model.classes)
		while(True):
			_, image = camera.read()
			tmp = image.shape
			img = [[]]
			img[0] = image.transpose(2, 0, 1)
			new = np.zeros((1, tmp[2], 224, 224))
			for i in range(tmp[2]):
				tmp = img[0][i]
				new[0][i] = np.resize(tmp, (224, 224))
				maximum, minimum = np.max(new[0][i]), np.min(new[0][i])
				new[0][i] = (new[0][i]-minimum)/(maximum-minimum)
			new = torch.from_numpy(new).type(torch.FloatTensor)
			#### to RGB
			#new = new[[2, 1, 0], ...]
			res = model.forward(torch.autograd.Variable(new))
			res = last_layer.forward(res)
			label = model.classes[res]
			cv2.putText(image, label, (10,500), font, 4, (1,1,1), 2, cv2.LINE_AA)
			cv2.imshow('preview', image)
			if cv2.waitKey(1) & 0xFF == ord('q'):
			    break

		camera.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	root = os.getcwd() + '/'
	# python3 train.py --data /tiny/imagenet/dir/ --save /dir/to/save/model/
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', metavar='DIR', default='model/my_model.pt',
	                    help='/dir/containing/model/')

	args = parser.parse_args()
	cam(args.model)