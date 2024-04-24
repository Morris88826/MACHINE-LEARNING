import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
try:
    from libs.dataloader import CustomMNIST
except:
    from dataloader import CustomMNIST
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.encoder = nn.Sequential(
            CNNBlock(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.encoder(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    data_path = "./data"
    
    train_image_path = os.path.join(data_path, "x_train_wr.npy")
    train_audio_path = os.path.join(data_path, "x_train_sp.npy")
    train_label_path = os.path.join(data_path, "y_train.csv")

    train_dataset = CustomMNIST(train_image_path, train_audio_path, train_label_path)
    split_ratio = 0.8
    train_size = int(split_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    image, audio, label = next(iter(val_loader))

    image_model = ImageNet()
    output = image_model(image)
    print(f"Output shape: {output.shape}")