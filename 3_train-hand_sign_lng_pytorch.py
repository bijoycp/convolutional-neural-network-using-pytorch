import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm


import torch.nn.functional as F

PATH_SAVE_MODEL_WEIGHT='model_weight/model_try.pth'

training_data = np.load("train_data.npy", allow_pickle=True)
print(len(training_data))

nb_classes=8

im_size=96

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 8, 3) # input is 1 image, 8 output channels, 3x3 kernel / window
        self.conv2 = nn.Conv2d(8, 16, 3) # input is 8, bc the first layer output 16. Then we say the output will be 32 channels, 3x3 kernel / window
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        # self.conv5 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(64*4*4, 512) #flattening.
        self.fc2 = nn.Linear(512, 128) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).
        self.fc3 = nn.Linear(128, 8)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = x.view(-1, 64*4*4) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        x = self.fc3(x)
        return F.softmax(x, dim=1)


net = Net()
print(net)


import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001) #0.001 ->0.00001
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,im_size,im_size)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)
print(val_size)


train_X = X[:-val_size]
train_y = y[:-val_size]


test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X), len(test_X))


BATCH_SIZE = 100
EPOCHS = 14

for epoch in range(EPOCHS):
    for i in range(0, len(train_X), BATCH_SIZE): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        # print(f"{i}:{i+BATCH_SIZE}")
        # print("loop")
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, im_size, im_size)
        batch_y = train_y[i:i+BATCH_SIZE]

        # print("size of batch x",batch_X.shape)
        # print("size of batch y",batch_y.shape)

        net.zero_grad()

        outputs = net(batch_X)
        # print(outputs)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()    # Does the update

    print(f"Epoch: {epoch}. Loss: {loss}")


    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(test_X)):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1, 1, im_size, im_size))[0]  # returns a list, 
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct/total, 3))

torch.save(net.state_dict(), PATH_SAVE_MODEL_WEIGHT)