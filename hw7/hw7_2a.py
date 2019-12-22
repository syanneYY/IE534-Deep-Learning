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
# from hw_7 import discriminator

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
# path = "visualization"
# os.mkdir(path)
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

print("model")
model = torch.load('cifar10.model')
model.cuda()
model.eval()
print("loaded model")


# Grab a sample batch from the test dataset. Create an alternative label which is simply +1 to the true label
batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

## save real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)

# Get the output from the fc10 layer and report the classification accuracy
_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

# Calculate the loss based on the alternative classes instead of the real classes.
## slightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

# save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)

# Modify the gradients to all be magnitude 1.0 or -1.0 and modify the original image.
# The example above changes each pixel by 10⁄255 based on the gradient sign.
# jitter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

## evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

## save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)