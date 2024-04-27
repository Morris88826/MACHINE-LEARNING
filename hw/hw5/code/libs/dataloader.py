import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


class CustomMNIST(Dataset):
    def __init__(self, image_path, audio_path, label_path=None):
        self.images = np.load(image_path)
        self.audios = np.load(audio_path)

        self.image_size = (28, 28)
        self.audio_size = (39, 13)

        if label_path is not None:
            self.labels = pd.read_csv(label_path)
            self.num_classes = len(self.labels.iloc[:, 1].unique())
        else:
            self.labels = None
            self.num_classes = 10 # default number of classes for MNIST dataset

        self.image_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(self.image_size[0], self.image_size[1])
        audio = self.audios[idx].reshape(self.audio_size[0], self.audio_size[1])

        image = self.image_tfm(image)
        if self.labels is not None:
            label = self.labels.iloc[idx, 1]
            return image, audio, label
        return image, audio

class Loader:
    def __init__(self, data_path, batch_size, split_ratio = 0.7) -> None:
        train_image_path = os.path.join(data_path, "x_train_wr.npy")
        train_audio_path = os.path.join(data_path, "x_train_sp.npy")
        train_label_path = os.path.join(data_path, "y_train.csv")

        train_dataset = CustomMNIST(train_image_path, train_audio_path, train_label_path)
        train_size = int(split_ratio * len(train_dataset))
        remaining_size = len(train_dataset) - train_size
        val_size = remaining_size//2
        test_size = remaining_size - val_size

        # set the seed for reproducibility
        torch.manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])
        print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

        # remove the seed
        torch.seed()
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    data_path = "./data"
    demo_path = "./demo"

    if not os.path.exists(demo_path):
        os.makedirs(demo_path)

    loader = Loader(data_path, batch_size=32)
    
    digits = defaultdict(list)
    for (image, audio, label) in loader.train_loader:
        image = image.numpy().squeeze()
        audio = audio.numpy().squeeze()
        label = label.numpy().squeeze().item()

        if len(digits[label]) < 5:
            digits[label].append((image, audio))

        if len(digits) == 10 and all(len(v) == 5 for v in digits.values()):
            break

    fig1, axs1 = plt.subplots(10, 5, figsize=(15, 20))
    fig2, axs2 = plt.subplots(10, 5, figsize=(15, 20))
    for i, (label, data) in enumerate(sorted(digits.items())):
        for j, (image, audio) in enumerate(data):
            axs1[label, j].imshow(image, cmap="gray")
            axs1[label, j].set_title(f"Label: {label}")
            axs1[label, j].axis("off")

            axs2[label, j].imshow(audio, aspect='auto', origin='lower')
            axs2[label, j].set_title(f"Label: {label}")
            # axs2[label, j].axis("off")

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(os.path.join(demo_path, "sample_image.png"))
    fig2.savefig(os.path.join(demo_path, "sample_audio.png"))
