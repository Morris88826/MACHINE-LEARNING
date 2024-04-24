import os
import torch
import shutil
import argparse
import numpy as np
from libs.model import ImageNet
from libs.dataloader import Loader
from torch.utils.tensorboard import SummaryWriter

def main(data_path, model_type, hyperparameters):

    if model_type == "Image":
        model = ImageNet()
    else:
        raise NotImplementedError("Model type not implemented")
    
    device = hyperparameters["device"]
    model.to(device)

    loader = Loader(data_path, hyperparameters["batch_size"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"])

    # checpoints
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    checkpoint_path = os.path.join(f'./checkpoints/{model_type}.pth')

    # Summary writer
    if not os.path.exists('./runs'):
        os.makedirs('./runs')
    summary_path = os.path.join('./runs', model_type)
    if os.path.exists(summary_path):    
        shutil.rmtree(summary_path)
    writer = SummaryWriter(log_dir=summary_path)


    best_accuracy = 0
    for epoch in range(hyperparameters["epochs"]):
        model.train()
        train_loss = []
        print(f"Epoch: {epoch}")
        for idx, (image, audio, label) in enumerate(loader.train_loader):
            image, audio, label = image.to(device), audio.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

            if idx % 100 == 0:
                print(f" - train_loss: {loss.item()}")

        train_loss = np.mean(train_loss)
        writer.add_scalar("Loss/train", train_loss, epoch)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = []
            for idx, (image, audio, label) in enumerate(loader.val_loader):
                image, audio, label = image.to(device), audio.to(device), label.to(device)
                output = model(image)
                loss = criterion(output, label)

                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)
    
            print(f"Validation Accuracy: {100 * correct / total}")
            
            print("=====================================")
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", 100 * correct / total, epoch)

            if 100 * correct / total > best_accuracy:
                best_accuracy = 100 * correct / total
                torch.save(model.state_dict(), checkpoint_path)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model on MNIST dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--model_type", type=str, default="Image", help="Type of model to train")
    args = parser.parse_args()

    if args.model_type not in ["Image", "Audio", "Hybrid"]:
        raise ValueError("Model type must be one of 'Image', 'Audio', 'Hybrid'")

    hyperparameters = {
        "lr": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    main(args.data_path, args.model_type, hyperparameters)