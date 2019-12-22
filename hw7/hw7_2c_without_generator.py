import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from hw_with1 import generator, discriminator
EXTRACT_FEATURE = 8
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__() #Specifying the model parameters
        #8 convolution layers with ReLU activation
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3,  padding = 1, stride = 1)
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
        # fc1 is considered the output of the critic which is a score determining of the input is real or a fake image coming from the generator
        self.fc1 = nn.Linear(196, 1)
        # fc10 is considered the auxiliary classifier which corresponds to the class label. Return both of these outputs in the forward call.
        self.fc10 = nn.Linear(196, 10)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

# LayerNormalization(normal_shape=normal_shape)
    def forward(self, x, extract_features=0): #Specifying the NN architecture
        if extract_features != 0:
            layers = [self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4,
                      self.conv_layer5, self.conv_layer6, self.conv_layer7, self.conv_layer8]
            output_dim = [32, 16, 16, 8, 8, 8, 8, 4]
            for i in range(extract_features):
                x = layers[i](x)
            x = F.max_pool2d(x, output_dim[extract_features - 1], output_dim[extract_features - 1])
            x = x.view(-1, 196)
            return x

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

        # if (extract_features == 8):
        #     x = F.max_pool2d(x, 4, 4)
        #     x = x.view(-1, 196)
        #     return x

        x = self.conv_layer5(x)
        x = self.layernorm_layer5(x)
        x = self.relu5(x)

        x = self.conv_layer6(x)
        x = self.layernorm_layer6(x)
        x = self.relu6(x)

        x = self.conv_layer7(x)
        x = self.layernorm_layer7(x)
        x = self.relu7(x)
        #


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
        self.BatchNorm = nn.BatchNorm2d(196)
        #
        # self.ReLU = F.relu()
        # self.Tanh = torch.Tanh()

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

        self.conv_layer8 = nn.Conv2d(196, 3, kernel_size=3, padding=1, stride=1)

        # self.apply(weights_init)


    def forward(self, x):

        x = self.fc1(x)
        x = x.view(-1, 196, 4, 4)
        x = self.BatchNorm(x)
        x = F.relu(x)

        x = self.conv_layer1(x)
        x = self.BatchNorm1(x)
        x = F.relu(x)

        x = self.conv_layer2(x)
        x = self.BatchNorm2(x)
        x = F.relu(x)

        x = self.conv_layer3(x)
        x = self.BatchNorm3(x)
        x = F.relu(x)

        x = self.conv_layer4(x)
        x = self.BatchNorm4(x)
        x = F.relu(x)

        x = self.conv_layer5(x)
        x = self.BatchNorm5(x)
        x = F.relu(x)

        x = self.conv_layer6(x)
        x = self.BatchNorm6(x)
        x = F.relu(x)

        x = self.conv_layer7(x)
        x = self.BatchNorm7(x)
        x = F.relu(x)

        x = self.conv_layer8(x)

        output = F.tanh(x)
        return output

batch_size = 128
# plot function, load discriminator model trained without the generator
def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
testloader = enumerate(testloader)

model = torch.load('cifar10.model')

model.cuda()
model.eval()


# Grab a sample batch from the test dataset.
batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model(X, EXTRACT_FEATURE)

    loss = -output.diag()
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_8.png', bbox_inches='tight')
plt.close(fig)