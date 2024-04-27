import torch.nn as nn
from .base import CNNBlock

class ImageNet_CNN(nn.Module):
    def __init__(self):
        super(ImageNet_CNN, self).__init__()
        self.name = "ImageNet_CNN"
        self.encoder = nn.Sequential(
            CNNBlock(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )
    
    def forward(self, x, embedding=False):
        x = x.view(-1, 1, 28, 28)
        x = self.encoder(x)
        x = self.fc(x)

        if embedding:
            return x
        
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    model = ImageNet_CNN()
    print(model)

    # save this into a image
