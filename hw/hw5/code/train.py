import os
import torch
import argparse
import numpy as np
from libs.dataloader import Loader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import get_hyperparameters, get_model, choose_model, model_names

def main(data_path, model_name, hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name)
    model.to(device)
    print("Training model: ", model_name)
    print("Hyperparameters: ", hyperparameters)

    loader = Loader(data_path, hyperparameters["batch_size"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # checpoints
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    checkpoint_path = os.path.join(f'./checkpoints/{model_name}.pth')

    best_accuracy = 0
    early_stop = 0
    for epoch in range(hyperparameters["epochs"]):
        model.train()
        train_loss = []
        print(f"Epoch: {epoch}")
        for idx, (image, audio, label) in enumerate(loader.train_loader):
            image = Variable(image).float().to(device)
            audio = Variable(audio).float().to(device)
            label = Variable(label).to(device)

            optimizer.zero_grad()

            if "Image" in model_name:
                output = model(image)
            elif "Audio" in model_name:
                output = model(audio)
            elif "Hybrid" in model_name:
                output = model(image, audio)
            else:
                raise ValueError(f"Invalid model name: {model_name}")

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

            if idx % 100 == 0:
                print(f" - train_loss: {loss.item()}")

        train_loss = np.mean(train_loss)
        print(f"Train Loss: {train_loss}")

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = []
            for idx, (image, audio, label) in enumerate(loader.val_loader):
                image = Variable(image).float().to(device)
                audio = Variable(audio).float().to(device)
                label = Variable(label).to(device)

                if "Image" in model_name:
                    output = model(image)
                elif "Audio" in model_name:
                    output = model(audio)
                elif "Hybrid" in model_name:
                    output = model(image, audio)
                else:
                    raise ValueError(f"Invalid model name: {model_name}")

                loss = criterion(output, label)

                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                val_loss.append(loss.item())
            val_loss = np.mean(val_loss)
            scheduler.step(val_loss)
            print(f"Validation Loss: {val_loss}")
            print(f"Validation Accuracy: {100 * correct / total}")
            
            print("=====================================")
            if 100 * correct / total > best_accuracy:
                best_accuracy = 100 * correct / total
                torch.save(model.state_dict(), checkpoint_path)
                early_stop = 0
            else:
                early_stop += 1

            if early_stop > 10:
                print("Early stopping")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model on MNIST dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--model_name", "-m", type=str, help="Name of model to train")
    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = choose_model()

    if args.model_name not in model_names:
        raise ValueError(f"Invalid model name: {args.model_name}")

    hyperparameters = get_hyperparameters(args.model_name)

    main(args.data_path, args.model_name, hyperparameters)
