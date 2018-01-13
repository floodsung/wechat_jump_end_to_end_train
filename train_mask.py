import os
import torch
import torch.nn as nn
from torch.autograd import Variable

import dataset84
import model
from unet import UNet,CNNEncoder

def main():

	# init conv net
	print("init net")

	unet = UNet(3,1)
	if os.path.exists("./unet.pkl"):
		unet.load_state_dict(torch.load("./unet.pkl"))
		print("load unet")
	unet.cuda()

	cnn = CNNEncoder()
	if os.path.exists("./cnn.pkl"):
		cnn.load_state_dict(torch.load("./cnn.pkl"))
		print("load cnn")
	cnn.cuda()

	# init dataset
	print("init dataset")
	data_loader = dataset84.jump_data_loader()

	# init optimizer
	unet_optimizer = torch.optim.Adam(unet.parameters(),lr=0.001)
	cnn_optimizer = torch.optim.Adam(cnn.parameters(),lr = 0.001)
	criterion = nn.MSELoss()

	# train
	print("training...")
	for epoch in range(1000):
		for i, (images, press_times) in enumerate(data_loader):
			images = Variable(images).cuda()
			press_times = Variable(press_times.float()).cuda()

			masks = unet(images)
			
			segmentations = images * masks
			predict_press_times = cnn(segmentations)

			loss = criterion(predict_press_times,press_times)

			unet_optimizer.zero_grad()
			cnn_optimizer.zero_grad()
			loss.backward()
			unet_optimizer.step()
			cnn_optimizer.step()

			if (i+1) % 10 == 0:
				print("epoch:",epoch,"step:",i,"loss:",loss.data[0])
			if (epoch+1) % 5 == 0 and i == 0:
				torch.save(unet.state_dict(),"./unet.pkl")
				torch.save(cnn.state_dict(),"./cnn.pkl")
				print("save model")

			

if __name__ == '__main__':
	main()