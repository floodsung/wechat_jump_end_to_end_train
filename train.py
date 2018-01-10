import os
import torch
import torch.nn as nn
from torch.autograd import Variable

import dataset

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer6 = nn.Linear(1600,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0),-1)
        out = self.layer6(out)

        return out # 64

def main():

	# init conv net
	print("init net")
	net = CNNEncoder()
	if os.path.exists("./model.pkl"):
		net.load_state_dict(torch.load("./model.pkl"))
		print("load model")

	net.cuda()

	# init dataset
	print("init dataset")
	data_loader = dataset.jump_data_loader()

	# init optimizer
	optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
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
				torch.save(net.state_dict(),"./model.pkl") #current is model.pkl
				print("save model")

if __name__ == '__main__':
	main()
