'''
Template for simple starts
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class CNNSimple(nn.Module):
    def __init__(self, in_shape=(28, 28, 1), filters=(32, 64, 128, 256), activation=F.relu, num_classes=10):
        super(CNNSimple, self).__init__()
        channels = in_shape[2]
        self.activation = activation
        self.conv1 = nn.Conv2d(channels,
                               filters[0],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True
                               )

        self.conv2 = nn.Conv2d(filters[0],
                               filters[1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True
                               )

        self.conv3 = nn.Conv2d(filters[1],
                               filters[2],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True
                               )

        self.conv4 = nn.Conv2d(filters[2],
                               filters[3],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True
                               )

        self.linear1 = nn.Linear(2 * 2 * filters[3], 256)

        self.linear2 = nn.Linear(256, num_classes)



    def forward(self, x):


        conv1 = self.activation(self.conv1(x))
        conv2 = self.activation(self.conv2(conv1))
        conv3 = self.activation(self.conv3(conv2))
        conv4 = self.activation(self.conv4(conv3))
        fc1 = self.activation(self.linear1(conv4.view(conv4.size(0), -1)))



        logits = self.activation(self.linear2(fc1))
        return logits, (conv1, conv2, conv3, conv4, fc1)


def SimpleModel(in_shape=(28, 28, 1), activation='relu', filters=(32, 64, 128, 256), num_classes=10):
    if activation == 'leaky_relu':
        activation = F.leaky_relu
    elif activation == 'prelu':
        activation = F.prelu
    else:
        activation = F.relu
    return CNNSimple(in_shape=in_shape, activation=activation, filters=filters, num_classes=num_classes)
