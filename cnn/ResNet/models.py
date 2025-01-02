import torch
import torch.nn as nn

"""
Implementation of the ResNet models.
"""

class block(nn.Module):
  def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
    super(block, self,).__init__()
    self.expansion = 4
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
    self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
    self.relu = nn.ReLU()
    self.identity_downsample = identity_downsample

  def forward(self, x):
    identity = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)

    # matching dimension of identity with output if dimension of identity and output don't match
    if self.identity_downsample is not None:
      identity = self.identity_downsample(identity)

    x += identity
    x = self.relu(x)
    return x


class ResNet(nn.Module):
  def __init__(self, block, layers, image_channels, num_classes):
    super(ResNet, self).__init__()
    self.in_channels = 64

    # conv1 layer: normal conv layer using the above block
    self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # Resnet layers
    self.layer1 = self.__make_layer__(block, layers[0], out_channels=64, stride=1) # conv2_x
    self.layer2 = self.__make_layer__(block, layers[1], out_channels=128, stride=2) # conv3_x
    self.layer3 = self.__make_layer__(block, layers[2], out_channels=256, stride=2) # conv4_x
    self.layer4 = self.__make_layer__(block, layers[3], out_channels=512, stride=2) # conv5_x

    self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # avg pool
    self.fc = nn.Linear(512*4, num_classes) # final fc layer to output class probability

  def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)

    return x

  def __make_layer__(self, block, num_residual_block, out_channels, stride):
    # num_residual_block = number of times the block is used while in the layer
    identity_downsample = None
    layers = []

    if stride != 1 or self.in_channels != out_channels*4:
      # if we change the number of channels
      identity_downsample = nn.Sequential(
          nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
          nn.BatchNorm2d(out_channels * 4)
        )

    # Layer that changes number of channels
    layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
    self.in_channels = out_channels * 4 # 256

    for i in range(1, num_residual_block):
      layers.append(block(self.in_channels, out_channels)) # 256 -> 64, 64*4 (256) again

    return nn.Sequential(*layers)


# ResNet50
def ResNet50(img_channels=3, num_classes=10):
  return ResNet(block, [3,4,6,3], img_channels, num_classes)

# ResNet101
def ResNet101(img_channels=3, num_classes=10):
  return ResNet(block, [3,4,23,3], img_channels, num_classes)

# ResNet152
def ResNet152(img_channels=3, num_classes=10):
  return ResNet(block, [3,8,36,3], img_channels, num_classes)
