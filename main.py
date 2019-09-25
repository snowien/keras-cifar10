import os

import argparse

from cifar10_resnet import ResNet


if __name__ == '__main__':
    model=ResNet()
    model.train()
