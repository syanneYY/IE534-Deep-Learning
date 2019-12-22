# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:05:28 2018

@author: Rachneet Kaur
"""
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__() #Specifying the model parameters
        #8 convolution layers with ReLU activation
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3,  padding = 1, stride = 1,)
        self.layernorm_layer1 = nn.LayerNorm(normalized_shape=[32, 32])
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_layer2 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3,  padding = 1, stride = 2)
        self.layernorm_layer2 = nn.LayerNorm(normalized_shape=[16, 16])
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_layer3 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride = 1)
        self.layernorm_layer3 = nn.LayerNorm(normalized_shape=[16, 16])
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_layer4 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride = 2)
        self.layernorm_layer4 = nn.LayerNorm(normalized_shape=[8, 8])
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_layer5 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride = 1)
        self.layernorm_layer5 = nn.LayerNorm(normalized_shape=[8, 8])
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_layer6 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride = 1)
        self.layernorm_layer6 = nn.LayerNorm(normalized_shape=[8, 8])
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_layer7 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride = 1)
        self.layernorm_layer7 = nn.LayerNorm(normalized_shape=[8, 8])
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_layer8 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1, stride = 2)
        self.layernorm_layer8 = nn.LayerNorm(normalized_shape=[4, 4])
        self.relu8 = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.MaxPool2d(4, stride=4)
        # fc1 is considered the output of the �critic� which is a score determining of the input is real or a fake image coming from the generator
        self.fc1 = nn.Linear(196, 1)
        # fc10 is considered the auxiliary classifier which corresponds to the class label. Return both of these outputs in the forward call.
        self.fc10 = nn.Linear(196, 10)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

# LayerNormalization(normal_shape=normal_shape)
    def forward(self, x): #Specifying the NN architecture

        x = self.conv_layer1(x)
        x = self.layernorm_layer1(x)
        x = self.relu1(x)

        x = self.conv_layer2(x)
        x = self.layernorm_layer2(x)
        x = self.relu2(x)

        x = self.conv_layer3(x)
        x = self.layernorm_layer3(x)
        x = self.relu3(x)

        x = self.conv_layer4(x)
        x = self.layernorm_layer4(x)
        x = self.relu4(x)

        x = self.conv_layer5(x)
        x = self.layernorm_layer5(x)
        x = self.relu5(x)

        x = self.conv_layer6(x)
        x = self.layernorm_layer6(x)
        x = self.relu6(x)

        x = self.conv_layer7(x)
        x = self.layernorm_layer7(x)
        x = self.relu7(x)

        x = self.conv_layer8(x)
        x = self.layernorm_layer8(x)
        x = self.relu8(x)

        x = self.pool(x)

        x = x.view(-1, 196)
        c = self.fc1(x)
        # c = self.softmax(c)
        s = self.fc10(x)
        # s = self.sigmoid(s)
        return c, s

"""GENERATOR NETWORK"""
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__() #Specifying the model parameters
        # fake image generated from the 100 dimensional input noise
        self.fc1 = nn.Linear(100, 196*4*4)
        self.BatchNorm = nn.BatchNorm2d(196*4*4)

        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh()

        self.conv_layer1 = nn.ConvTranspose2d(196, 196, kernel_size=4,  padding=1, stride=2)
        self.BatchNorm1 = nn.BatchNorm2d(196)

        self.conv_layer2 = nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1)
        self.BatchNorm2 = nn.BatchNorm2d(196)

        self.conv_layer3 = nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1)
        self.BatchNorm3 = nn.BatchNorm2d(196)

        self.conv_layer4 = nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1)
        self.BatchNorm4 = nn.BatchNorm2d(196)

        self.conv_layer5 = nn.ConvTranspose2d(196, 196, kernel_size=4, padding=1, stride=2)
        self.BatchNorm5 = nn.BatchNorm2d(196)

        self.conv_layer6 = nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1)
        self.BatchNorm6 = nn.BatchNorm2d(196)

        self.conv_layer7 = nn.ConvTranspose2d(196, 196, kernel_size=4, padding=1, stride=2)
        self.BatchNorm7 = nn.BatchNorm2d(196)

        self.conv_layer8 = nn.Conv2d(196, 196, kernel_size=3, padding=1, stride=1)

        # self.apply(weights_init)


    def forward(self, input):
        x = self.fc1(input)
        x = self.BatchNorm(x)
        x = self.ReLU(x)

        x = self.conv_layer1(input)
        x = self.BatchNorm1(x)
        x = self.ReLU(x)

        x = self.conv_layer2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU(x)

        x = self.conv_layer3(x)
        x = self.BatchNorm3(x)
        x = self.ReLU(x)

        x = self.conv_layer4(x)
        x = self.BatchNorm4(x)
        x = self.ReLU(x)

        x = self.conv_layer5(x)
        x = self.BatchNorm5(x)
        x = self.ReLU(x)

        x = self.conv_layer6(x)
        x = self.BatchNorm6(x)
        x = self.ReLU(x)

        x = self.conv_layer7(x)
        x = self.BatchNorm7(x)
        x = self.ReLU(x)

        x = self.conv_layer8(x)

        output = self.Tanh(x)
        return output


# def main():
batch_size = 128
"""the train and test loader"""
# Notice the normalize function has mean and std values of 0.5. The original dataset is scaled between 0 and 1.
# These values will normalize all of the inputs to between -1 and 1 which matches the hyperbolic tangent function from the generator.
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)



"""MODEL"""
model = discriminator()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
'''
Train Mode
Create the batch -> Zero the gradients -> Forward Propagation -> Calculating the loss 
-> Backpropagation -> Optimizer updating the parameters -> Prediction 
'''

start_time = time.time()
train_accuracy = []
learning_rate = 0.0001
for epoch in range(200):  # loop over the dataset multiple times
    if (epoch == 50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / 10.0
    if (epoch == 75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / 100.0
    # optimizer = optim.Adam(model.parameters(), lr=LR) #ADAM optimizer
    running_loss = 0.0

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if (Y_train_batch.shape[0] < batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
        if (epoch> 6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'step' in state.keys():
                        if (state['step'] >= 1024):
                            state['step'] = 1000
        loss.backward()

        optimizer.step()
        prediction = output.data.max(1)[1]  # Label Prediction
        accuracy = (float(prediction.eq(Y_train_batch.data).sum()) / float(batch_size)) * 100.0  # Computing the training accuracy
        train_accuracy.append(accuracy)
    accuracy_epoch = np.mean(train_accuracy)
    print('\nIn epoch ', epoch, ' the accuracy of the training set =', accuracy_epoch)
    # for i, batch in enumerate(data_train, 0):
    #     data, target = batch
    #     data, target = Variable(data).cuda(), Variable(target).cuda()
    #     optimizer.zero_grad() #Zero the gradients at each epoch
    #     output = model(data)#Forward propagation
    #     #Negative Log Likelihood Objective function
    #     loss = loss_func(output, target)
    #     loss.backward() #Backpropagation
    #     optimizer.step() #Updating the parameters using ADAM optimizer
    #     prediction = output.data.max(1)[1] #Label Prediction
    #     accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0 #Computing the training accuracy
    #     train_accuracy.append(accuracy)
    # accuracy_epoch = np.mean(train_accuracy)
    # print('\nIn epoch ', epoch,' the accuracy of the training set =', accuracy_epoch)
end_time = time.time()

#torch.save(model.state_dict(), 'params_cifar10_dcnn_LR001.ckpt') #To save the trained model
'''
Calculate accuracy of trained model on the Test Set
Create the batch ->  Forward Propagation -> Prediction 
'''
# correct = 0
# total = 0
test_accuracy = []

#Extra Credit Comparision of Heuristic and Monte Carlo method
# if heuristic:
#     MonteCarlo = 1 #Only one itertion if we are using the heuristic
#     #Comparing the accuarcy of the heuristic and Monte Carlo Method
model.eval()
for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):

    if (Y_test_batch.shape[0] < batch_size):
        continue

    X_test_batch = Variable(X_test_batch).cuda()
    Y_test_batch = Variable(Y_test_batch).cuda()
    _, output = model(X_test_batch)
    prediction = output.data.max(1)[1]  # Label Prediction
    accuracy = (float(prediction.eq(Y_test_batch.data).sum()) / float(batch_size)) * 100.0  # Computing the training accuracy
    test_accuracy.append(accuracy)
accuracy_epoch_test = np.mean(test_accuracy)
print(' the accuracy of the testing set =', accuracy_epoch_test)

torch.save(model,'cifar10_hw7.model')

# if __name__ == '__main__':
#     main()
