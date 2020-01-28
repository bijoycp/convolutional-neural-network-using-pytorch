import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm


import torch.nn.functional as F

IMG_SIZE = 96
LR = 1e-3

nb_classes=15

MODEL_NAME = 'model_15.pth'



class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 8, 3) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(8, 16, 3) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        # self.conv5 = nn.Conv2d(64, 128, 3)

        x = torch.randn(96,96).view(-1,1,96,96)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 128) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).
        self.fc3 = nn.Linear(128, nb_classes)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            print(self._to_linear)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        x = self.fc3(x)
        return F.softmax(x, dim=1)


net = Net()

print(net)

net.load_state_dict(torch.load(MODEL_NAME))
net.eval()

# organize imports
import cv2
import imutils
import numpy as np

from collections import Counter

import time



            #0    1    2     3    4      5    6    7        8     9    10   11   12   13   14    15    16   17   18   19   20   21   22   23   24    25    26   27  28
# out_label=['blnk', 'W', 'E', 'R', 'BKSP', 'Q', 'D', 'O', 'A', 'SPC', 'I', 'F', 'O', 'C', 'W', 'Y', 'BLNK', 'V', 'H', 'H', 'P', 'A', 'S', 'L', 'K', 'X', 'N', 'B', ]

out_label={0: 'H', 1: 'I', 2: 'bsp', 3: 'R', 4: 'C', 5: 'L', 6: 'W', 7: 'spc', 8: 'B', 9: 'D', 10: 'E', 11: 'blnk', 12: 'O', 13: 'P', 14: 'A'}

pre=[]

s=''
cchar=[0,0]
c1=''

# initialize weight for running average
aWeight = 0.5

# get the reference to the webcam
camera = cv2.VideoCapture(0)

# region of interest (ROI) coordinates
top, right, bottom, left = 170, 150, 425, 450

# initialize num of frames
num_frames = 0

flag=0
flag1=0

# keep looping, until interrupted
while(True):
    # get the current frame
    (grabbed, frame) = camera.read()

    # resize the frame
    frame = imutils.resize(frame, width=700)

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()

    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    roi = frame[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    
    # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
    
    img=gray
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img=cv2.imread("240fn.jpg",cv2.IMREAD_GRAYSCALE)
    # img=cv2.cvtColor(bw_image,cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    test_data =img

    orig = img
    data = torch.Tensor(img).view(-1, 1, 96, 96)


    net_out = net(data)  # returns a list,
    # print("net_out",net_out) 
    predicted_class = torch.argmax(net_out)

    # print("predicted_class",predicted_class)


    
    pnb=predicted_class.item()
    # print("pnb",pnb)
    # print(str(np.argmax(predicted_class))+" "+str(out_label[pnb]))

    pre.append(out_label[pnb]) 
    if out_label[pnb]=='blnk':
        pass
    else:
        print("pnb",pnb)
        print("output: ",out_label[pnb])

    


    cv2.putText(clone,
           '%s ' % (str(out_label[pnb])),
           (450, 150), cv2.FONT_HERSHEY_PLAIN,5,(0, 255, 0))

            

            
        


    # draw the segmented hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

    cv2.putText(clone,
                   '%s ' % (str(s)),
                   (10, 60), cv2.FONT_HERSHEY_PLAIN,3,(0, 0, 0))

    # increment the number of frames
    num_frames += 1
    # time.sleep(.3)
    # display the frame with segmented hand
    cv2.imshow("Video Feed", clone)

    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF

    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break

    elif keypress == 27:
        break

# free up memory
camera.release()
cv2.destroyAllWindows()
