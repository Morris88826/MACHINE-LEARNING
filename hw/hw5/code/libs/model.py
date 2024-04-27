import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
    
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
    
class AudioNet(nn.Module):
    def __init__(self, embedding_size, hidden_size=64, num_layers=2, bidirectional=True):
        super(AudioNet, self).__init__()
        
        # bidirectional LSTM
        self.encoder = nn.Sequential(
            nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional),
        )
        self.bidirectional = bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

        self.flatten = nn.Flatten()

    
    def forward(self, x):
        x, (hn, _) = self.encoder(x)
        if self.bidirectional:
            forward_hn = hn[-2] # forward direction of the last layer
            backward_hn = hn[-1] # backward direction of the last layer
            x = torch.cat([forward_hn, backward_hn], dim=1)
        else:
            x = hn[-1]
        x = self.classifier(x)
        return x
    
class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        self.image_encoder = nn.Sequential(
            CNNBlock(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.audio_encoder = nn.Sequential(
            CNNBlock(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.image_clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.audio_clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 9, 128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )
    
    def forward(self, image, audio):
        image = image.view(-1, 1, 28, 28)
        audio = audio.view(-1, 1, 39, 13)

        x1 = self.image_encoder(image)
        x1 = self.image_clf(x1)

        x2 = self.audio_encoder(audio)
        x2 = self.audio_clf(x2)

        x = self.classifier(torch.cat([x1, x2], dim=1))
        return x

    
