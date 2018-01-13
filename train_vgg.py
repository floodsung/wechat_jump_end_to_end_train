import os
import torch
import torch.nn as nn
from torch.autograd import Variable

import dataset
import vgg

def main():

	# init conv net
	print("init net")
	net = vgg.vgg11()
	if os.path.exists("./model.pkl"):
		net.load_state_dict(torch.load("./model.pkl"))
		print("load model")

	net.cuda()

	# init dataset
	print("init dataset")
	data_loader = dataset.jump_data_loader()

	# init optimizer
	optimizer = torch.optim.Adam(net.parameters(),lr=0.0001)
	criterion = nn.MSELoss()

	# train
	print("training...")
	for epoch in range(1000):
		for i, (images, press_times) in enumerate(data_loader):
			images = Variable(images).cuda()
			press_times = Variable(press_times.float()).cuda()

			predict_press_times = net(images)

			loss = criterion(predict_press_times,press_times)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i+1) % 10 == 0:
				print("epoch:",epoch,"step:",i,"loss:",loss.data[0])
			if (epoch+1) % 5 == 0 and i == 0:
				torch.save(net.state_dict(),"./vgg.pkl")
				print("save model")

if __name__ == '__main__':
	main()