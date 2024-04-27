import os
import torch
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from libs.dataloader import Loader
from config import get_model, choose_model
from sklearn.metrics import f1_score

def main(data_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join('./checkpoints', f'{model_name}.pth')
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Model checkpoint not found at {ckpt_path}")
    
    model = get_model(model_name)
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    print(f"Load model from {ckpt_path} successfully")

    loader = Loader(data_path, batch_size=32)

    with torch.no_grad():
        predictions = []
        gt = []
        for _, (image, audio, label) in enumerate(loader.test_loader):
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

            _, pred = torch.max(output, 1)
            predictions.extend(pred.cpu().numpy())
            gt.extend(label.cpu().numpy())

    predictions = np.array(predictions)
    gt = np.array(gt)
    accuracy = np.mean(predictions == gt)
    f1 = f1_score(gt, predictions, average='macro')

    print(f"Model: {model_name}")
    print(f" - accuracy: {accuracy*100:.2f}%")
    print(f" - f1 score: {f1:.4f}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on MNIST dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--model_name", "-m", type=str, help="Name of model to use")
    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = choose_model()
    main(args.data_path, args.model_name)
    