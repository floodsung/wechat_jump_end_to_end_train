# coding: utf-8
import os
import sys
import subprocess
import time
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from unet import UNet,CNNEncoder


def pull_screenshot(name="autojump.png"):
    process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
    screenshot = process.stdout.read()
    f = open(name, 'wb')
    f.write(screenshot)
    f.close()

def preprocess(image):
    w, h = image.size
    top =  (h - w)/2

    image = image.crop((0,top,w,w+top))
    image = image.convert('RGB')
    image = image.resize((84,84), resample=Image.LANCZOS)

    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform = transforms.Compose([transforms.ToTensor(),normalize])

    image = transform(image)

    return image

def set_button_position(im):
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    w, h = im.size
    print(w,h)
    left = int(w / 2)
    top = int(1584 * (h / 1920.0))
    left = int(random.uniform(left-50, left+50))
    top = int(random.uniform(top-10, top+10))    
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top

def jump(press_time):
    press_time = int(press_time)
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=swipe_x1,
        y1=swipe_y1,
        x2=swipe_x2,
        y2=swipe_y2,
        duration=press_time
    )
    os.system(cmd)


def main():

    # init conv net
    
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

    unet.eval()
    cnn.eval()
    
    print("load ok")

    while True:
        pull_screenshot("autojump.png") # obtain screen and save it to autojump.png
        image = Image.open('./autojump.png')
        set_button_position(image)
        image = preprocess(image)
        
        image = Variable(image.unsqueeze(0)).cuda()
        mask = unet(image)

        plt.imshow(mask.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='hot', interpolation='nearest')
        plt.show()
        
        segmentation = image * mask

        press_time = cnn(segmentation)
        press_time = press_time.cpu().data[0].numpy()
        print(press_time)
        jump(press_time)
        
        time.sleep(random.uniform(0.6, 1.1))






if __name__ == '__main__':
    main()