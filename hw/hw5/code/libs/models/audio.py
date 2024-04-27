import torch
import torch.nn as nn
from .base import CNNBlock

class AudioNet_CNN(nn.Module):
    def __init__(self):
        super(AudioNet_CNN, self).__init__()
        self.name = "AudioNet_CNN"
        self.encoder = nn.Sequential(
            CNNBlock(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 9, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x, embedding=False):
        x = x.view(-1, 1, 39, 13)
        x = torch.transpose(x, 2, 3)
        x = self.encoder(x)
        x = self.fc(x)
        if embedding:
            return x
        
        x = self.classifier(x)
        return x

class AudioNet_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size=64, num_layers=2, bidirectional=False):
        super(AudioNet_LSTM, self).__init__()
        self.name = "AudioNet_{}LSTM".format("Bi" if bidirectional else "")
        # bidirectional LSTM
        self.encoder = nn.Sequential(
            nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional),
        )
        self.bidirectional = bidirectional
        self.factor = 2 if bidirectional else 1

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.factor, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 10)
        )

    
    def forward(self, x, embedding=False):
        x, (hn, _) = self.encoder(x)
        if self.bidirectional:
            forward_hn = hn[-2] # forward direction of the last layer
            backward_hn = hn[-1] # backward direction of the last layer
            x = torch.cat([forward_hn, backward_hn], dim=1)
        else:
            x = hn[-1]
        
        x = self.fc(x)
        
        if embedding:
            return x

        x = self.classifier(x)
        return x