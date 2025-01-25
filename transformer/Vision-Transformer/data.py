from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

batch_size = 128

transform = ToTensor()

data_path = "/home/ushtro/Documents/AI_ML/MyWork/computer-vision/data"
train_data = MNIST(root=data_path, train=True, download =True, transform=transform)
test_data = MNIST(root=data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

print("Data ready!")
