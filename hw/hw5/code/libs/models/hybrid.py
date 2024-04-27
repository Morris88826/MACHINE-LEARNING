import torch
import torch.nn as nn
from .base import CNNBlock

class HybridNet_CNN(nn.Module):
    def __init__(self):
        super(HybridNet_CNN, self).__init__()
        self.name = "HybridNet_CNN"

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
    
    def forward(self, image, audio, embedding=False):
        image = image.view(-1, 1, 28, 28)
        audio = audio.view(-1, 1, 39, 13)

        x1 = self.image_encoder(image)
        x1 = self.image_clf(x1)

        x2 = self.audio_encoder(audio)
        x2 = self.audio_clf(x2)

        if embedding:
            return x1, x2
        
        x = self.classifier(torch.cat([x1, x2], dim=1))
        return x


class HybridNet_CNN_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size=64, num_layers=2, bidirectional=False):
        super(HybridNet_CNN_LSTM, self).__init__()
        self.name = "HybridNet_CNN_LSTM"

        self.image_encoder = nn.Sequential(
            CNNBlock(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.bidirectional = bidirectional
        self.audio_encoder = nn.Sequential(
            nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional),
        )

        self.image_clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.factor = 2 if bidirectional else 1
        self.audio_clf = nn.Sequential(
            nn.Linear(hidden_size * self.factor, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )
    
    def forward(self, image, audio, embedding=False):
        image = image.view(-1, 1, 28, 28)

        x1 = self.image_encoder(image)
        x1 = self.image_clf(x1)
    
        x2,(hn, _)  = self.audio_encoder(audio)
        if self.bidirectional:
            forward_hn = hn[-2] # forward direction of the last layer
            backward_hn = hn[-1] # backward direction of the last layer
            x2 = torch.cat([forward_hn, backward_hn], dim=1)
        else:
            x2 = hn[-1]
        
        x2 = self.audio_clf(x2)
        if embedding:
            return x1, x2
        
        x = self.classifier(torch.cat([x1, x2], dim=1))
        return x
