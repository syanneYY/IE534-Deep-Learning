""""Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset.
The convolution network should use (A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation.
For 10% extra credit, compare dropout test accuracy (i) using the heuristic prediction rule and (ii) Monte Carlo simulation.
For full credit, the model should achieve 80-90% Test Accuracy.
Submit via Compass (1) the code and (2) a paragraph (in a PDF document) which reports the results and briefly describes the model architecture.
"""
# Convolution layer 1: 64 channels, k = 4,s = 1, P = 2.
# Batch normalization
# Convolution layer 2: 64 channels, k = 4,s = 1, P = 2.
# Max Pooling: s = 2, k = 2.
# Dropout
# Convolution layer 3: 64 channels, k = 4,s = 1, P = 2.
# Batch normalization
# Convolution layer 4: 64 channels, k = 4,s = 1, P = 2.
# Max Pooling → Dropout
#
#
# Convolution layer 5: 64 channels, k = 4,s = 1, P = 2.
# Batch normalization
# Convolution layer 6: 64 channels, k = 3,s = 1, P = 0.
# Dropout
# Convolution layer 7: 64 channels, k = 3,s = 1, P = 0.
# Batch normalization
# Convolution layer 8: 64 channels, k = 3,s = 1, P = 0.
# Batch normalization, Dropout
# Fully connected layer 1: 500 units.
# Fully connected layer 2: 500 units.
# Linear → Softmax function
#

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torch.utils


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
        # Conv block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    #             nn.Dropout2d(p=0.5),

            # Conv block 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv block 3
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv block 3
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
    #             nn.Dropout2d(p=0.5)
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
#             nn.ReLU(inplace=True), since this is last layer, no relu
        )

#         self.conv_layer = nn.Sequential(
#             # Conv block 1
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
# #             nn.Dropout2d(p=0.5),

#             # Conv block 2
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.05),

#             # Conv block 3
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
# #             nn.Dropout2d(p=0.5),

           
#         )

#         self.fc_layer = nn.Sequential(
#             nn.Dropout(p=0.1),
#             nn.Linear(4096, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
#             nn.Linear(512,10)
# #             nn.ReLU(inplace=True), since this is last layer, no relu
#         )

            

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
                

    def forward(self, x):
        x = self.conv_layer(x)

        x = x.view(x.size(0), -1)

        x = self.fc_layer(x)

        return x




def main():
    net = CNN()
    net = net.cuda()
    ## Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    # net = net().cuda
    ## GPU????
    # if opt.is_gpu:
    #     net = net.cuda()
    #     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #     cudnn.benchmark = True






    B_S = 256
    ## Train dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    ## Test dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
         root='./data',
         train=True, download=False, transform=transform_train)#root=dataroot,
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=B_S, shuffle=True, num_workers=2) #32

    test_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=False, download=False, transform=transform_test) #root=dataroot,
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=B_S, shuffle=False, num_workers=2) #32

    #
    # def accuracy(output, target, topk=(1,)):
    #     """Computes the accuracy over the k top predictions for the specified values of k"""
    #     with torch.no_grad():
    #         maxk = max(topk)
    #         batch_size = target.size(0)
    #
    #         _, pred = output.topk(maxk, 1, True, True)
    #         pred = pred.t()
    #         correct = pred.eq(target.view(1, -1).expand_as(pred))
    #
    #         res = []
    #         for k in topk:
    #             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #             res.append(correct_k.mul_(100.0 / batch_size))
    #         return res



    ## Train
    for epoch in range(60):
        scheduler.step()
        each_loss = 0.0
#         all_correct = 0
        correct = 0
        total = 0
#         all_data = 0
        for i, data in enumerate(train_loader, start=0):
            # if i > 500:
            #     break
            X_in, Y = data
            inputs = X_in.cuda()
            labels = Y.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if epoch > 16:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if state['step'] >= 1024:
                            state['step'] = 1000
            optimizer.step()

            each_loss += loss.data[0]
            _, pred = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
#             pred = pred.t()
#             correct = pred.eq(labels.view(1, -1).expand_as(pred))
#             correct = correct[:].view(-1).float().sum(0, keepdim=True)
            # print(correct)
            # res.append(correct.mul_(100.0 / B_S))
#             all_correct += correct
#             all_data += X_in.size(0)
            # print(all_correct/all_data)
            # print(
            #     "Iteration: {0} | Loss: {1} | Training accuracy: {2}% ".format(epoch + 1, each_loss, res))
        # Normalizing the loss by the total number of train batches
        each_loss /= len(train_loader)
        """accuracy"""
        acc = correct/total
        print(
            "Iteration: {0} | Loss: {1} | Training accuracy: {2}% | LR: {3}%".format(epoch + 1, each_loss, acc, optimizer.param_groups[-1]['lr']))

    # print(res)
    print('==> Finished Training ...')


    #     prediction,_ = torch.max(outputs, dim = 1)
    #     if (prediction == labels):
    #         total_correct += 1
    # print(total_correct)
    # print(total_correct / np.float(len(x_train)))
    #     trn_accuracy = accuracy(outputs, )
    #     tst_accuracy = calculate_accuracy(testloader)
    #     print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(epoch + 1, each_loss,
    #                                                                                               trn_accuracy,
    #                                                                                               tst_accuracy))

    #     # save model
    #     if epoch % 50 == 0:
    #         print('==> Saving model ...')
    #         state = {
    #             'net': net.module if opt.is_gpu else net,
    #             'epoch': epoch,
    #         }
    #         if not os.path.isdir('checkpoint'):
    #             os.mkdir('checkpoint')
    #         torch.save(state, '../checkpoint/ckpt.t7')
    #
    # print('==> Finished Training ...')
    
    net = net.eval()
    with torch.no_grad():
        test_correct = 0
        total = 0
        for i, data in enumerate(test_loader, start=0):
            # if i >64:
            #     break
            X_in, Y = data
            inputs = X_in.cuda()
            labels = Y.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            _, pred = torch.max(outputs, dim=1)
            total += labels.size(0)
            test_correct += (pred == labels).sum().item()
    test_acc = test_correct/total
    print("Test accuracy: {0}%".format(test_acc))

if __name__ == '__main__':
    main()
