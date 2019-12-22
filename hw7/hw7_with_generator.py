import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
import os

path = "output"
os.mkdir(path)
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

print("run hw_with.py")
# gradient penalty described in the Wasserstein GAN section
def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    # plot a 10 by 10 grid of images scaled between 0 and 1
    # and add noise
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


    aD = discriminator()
    aD.cuda()

    aG = generator()
    aG.cuda()

    optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

    criterion = nn.CrossEntropyLoss()


    n_z = 100
    n_classes = 10
    # a random batch of noise for the generator
    np.random.seed(352)
    label = np.asarray(list(range(10))*10)
    noise = np.random.normal(0,1,(100,n_z))
    label_onehot = np.zeros((100,n_classes))
    label_onehot[np.arange(100), label] = 1
    noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
    noise = noise.astype(np.float32)

    save_noise = torch.from_numpy(noise)
    save_noise = Variable(save_noise).cuda()



    num_epochs = 200
    gen_train = 5

    # before epoch training loop starts
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []


    # Train the model
    for epoch in range(0,num_epochs):
        if (epoch > 6):
            for group in optimizer_d.param_groups:
                for p in group['params']:
                    state = optimizer_d.state[p]
                    if 'step' in state.keys():
                        if (state['step'] >= 1024):
                            state['step'] = 1000
        if (epoch > 6):
            for group in optimizer_g.param_groups:
                for p in group['params']:
                    state = optimizer_g.state[p]
                    if 'step' in state.keys():
                        if (state['step'] >= 1024):
                            state['step'] = 1000
        aG.train()
        aD.train()
        start_time = time.time()
        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

            if(Y_train_batch.shape[0] < batch_size):
                continue

            # train G
            if ((batch_idx % gen_train) == 0):
                for p in aD.parameters():
                    p.requires_grad_(False)

                aG.zero_grad()

                label = np.random.randint(0, n_classes, batch_size)
                noise = np.random.normal(0, 1, (batch_size, n_z))
                label_onehot = np.zeros((batch_size, n_classes))
                label_onehot[np.arange(batch_size), label] = 1
                noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
                noise = noise.astype(np.float32)
                noise = torch.from_numpy(noise)
                noise = Variable(noise).cuda()
                fake_label = Variable(torch.from_numpy(label)).cuda()

                fake_data = aG(noise)
                gen_source, gen_class = aD(fake_data)

                gen_source = gen_source.mean()
                gen_class = criterion(gen_class, fake_label)

                gen_cost = -gen_source + gen_class
                gen_cost.backward()

                optimizer_g.step()

            # train D
            for p in aD.parameters():
                p.requires_grad_(True)

            aD.zero_grad()

            # train discriminator with input from generator
            label = np.random.randint(0, n_classes, batch_size)
            noise = np.random.normal(0, 1, (batch_size, n_z))
            label_onehot = np.zeros((batch_size, n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()
            with torch.no_grad():
                fake_data = aG(noise)

            disc_fake_source, disc_fake_class = aD(fake_data)

            disc_fake_source = disc_fake_source.mean()
            disc_fake_class = criterion(disc_fake_class, fake_label)

            # train discriminator with input from the discriminator
            real_data = Variable(X_train_batch).cuda()
            real_label = Variable(Y_train_batch).cuda()
            disc_real_source, disc_real_class = aD(real_data)

            prediction = disc_real_class.data.max(1)[1]
            accuracy = (float(prediction.eq(real_label.data).sum()) / float(batch_size)) * 100.0

            disc_real_source = disc_real_source.mean()
            disc_real_class = criterion(disc_real_class, real_label)
            # print()
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)

            disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
            disc_cost.backward()

            optimizer_d.step()

            # within the training loop
            loss1.append(gradient_penalty.item())
            loss2.append(disc_fake_source.item())
            loss3.append(disc_real_source.item())
            loss4.append(disc_real_class.item())
            loss5.append(disc_fake_class.item())
            acc1.append(accuracy)
            if ((batch_idx % 50) == 0):
                print(epoch, batch_idx, "%.2f" % np.mean(loss1),
                      "%.2f" % np.mean(loss2),
                      "%.2f" % np.mean(loss3),
                      "%.2f" % np.mean(loss4),
                      "%.2f" % np.mean(loss5),
                      "%.2f" % np.mean(acc1))


        # Test the model
        aD.eval()
        with torch.no_grad():
            test_accu = []
            for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
                X_test_batch, Y_test_batch = Variable(X_test_batch).cuda(), Variable(Y_test_batch).cuda()

                with torch.no_grad():
                    _, output = aD(X_test_batch)

                prediction = output.data.max(1)[1]  # first column has actual prob.
                accuracy = (float(prediction.eq(Y_test_batch.data).sum()) / float(batch_size)) * 100.0
                test_accu.append(accuracy)
                accuracy_test = np.mean(test_accu)
        print('Testing', accuracy_test, time.time() - start_time)

        ### save output
        with torch.no_grad():
            aG.eval()
            samples = aG(save_noise)
            samples = samples.data.cpu().numpy()
            samples += 1.0
            samples /= 2.0
            samples = samples.transpose(0, 2, 3, 1)
            aG.train()

        fig = plot(samples)
        plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
        plt.close(fig)

        if (((epoch + 1) % 1) == 0):
            torch.save(aG, 'tempG.model')
            torch.save(aD, 'tempD.model')

    torch.save(aG,'generator.model')
    torch.save(aD,'discriminator.model')


if __name__ == '__main__':
    main()
