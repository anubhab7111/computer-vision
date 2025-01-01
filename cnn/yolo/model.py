import torch
import torch.nn as nn

class YOLOv1(nn.Module):
  def __init__(self, S = 7, B = 2, C = 20):
    super(YOLOv1, self).__init__()
    self.S, self.B, self.C = S, B, C
    self.depth = B * 5 + C
    depth = self.depth


    # Feature extractor
    layers = [
        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
        nn.LeakyReLU(negative_slope = 0.1),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(64, 192, kernel_size = 3, stride = 1, padding = 1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(192, 128, kernel_size = 1, stride = 1, padding = 0),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(256, 256, kernel_size = 1, stride = 1, padding = 0),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.MaxPool2d(kernel_size=2,stride=2)
    ]

    for i in range(4):
      layers += [
          nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0),
          nn.LeakyReLU(negative_slope=0.1),
          nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
          nn.LeakyReLU(negative_slope=0.1),
      ]

    layers += [
        nn.Conv2d(512, 512, kernel_size = 1, stride = 1, padding = 1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 0),
        nn.LeakyReLU(negative_slope=0.1),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ]

    for i in range(2):
      layers += [
          nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0),
          nn.LeakyReLU(negative_slope=0.1),
          nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1),
          nn.LeakyReLU(negative_slope=0.1),
      ]

    layers += [
        nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(1024, 1024, kernel_size = 3, stride = 2, padding = 1),
        nn.LeakyReLU(negative_slope=0.1),
    ]

    layers += [
        nn.Conv2d(1024, 1024, kernel_size = 3, stride = 1, padding = 1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(1024, 1024, kernel_size = 3, stride = 1, padding = 1),
        nn.LeakyReLU(negative_slope=0.1),
    ]

    # Fully Connected Layers:
    layers += [
        nn.Flatten(),
        nn.Linear(S * S * 1024, 4096),
        nn.Dropout(),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(4096, S * S * depth),
    ]

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    x = self.model(x)
    return x.view(-1, self.S, self.S, self.depth) #reshaping output
