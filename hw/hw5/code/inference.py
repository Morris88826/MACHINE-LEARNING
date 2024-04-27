import os
import shutil
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from libs.dataloader import CustomMNIST
from config import get_model, choose_model

def main(data_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join('./checkpoints', f'{model_name}.pth')
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Model checkpoint not found at {ckpt_path}")

    model = get_model(model_name)
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    print("Load model from {} successfully".format(ckpt_path))

    test_image_path = os.path.join(data_path, "x_test_wr.npy")
    test_audio_path = os.path.join(data_path, "x_test_sp.npy")
    test_dataset = CustomMNIST(test_image_path, test_audio_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for idx, (image, audio) in enumerate(test_loader):
            image = Variable(image).float().to(device)
            audio = Variable(audio).float().to(device)

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

    submission = pd.DataFrame({"row_id": np.arange(len(predictions)), "label": predictions})
    
    if not os.path.exists("./results"):
        os.makedirs("./results")
    output_path = os.path.join("./results", "Mu-Ruei_Tseng_preds_{}.csv".format(model_name))
    submission.to_csv(output_path, index=False)
    print("Save predictions to {}".format(output_path))
    return output_path

def visualize(data_path, csv_path, model_name):
    test_image_path = os.path.join(data_path, "x_test_wr.npy")
    test_images = np.load(test_image_path).reshape(-1, 28, 28)

    preds = pd.read_csv(csv_path)["label"].values

    random_idx = np.random.choice(len(test_images), 5)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, idx in enumerate(random_idx):
        axes[i].imshow(test_images[idx], cmap="gray")
        axes[i].set_title(f"Prediction: {preds[idx]}")
        axes[i].axis("off")
    
    plt.suptitle(f"Predictions for {model_name} model")
    plt.tight_layout()
    plt.savefig(f"./demo/pred_{model_name}.png")
    plt.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference the model on MNIST dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--model_name", "-m", type=str, help="Type of model to use")
    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = choose_model()

    output_path = main(args.data_path, args.model_name)

    ##############################
    best_result = "./results/Mu-Ruei_Tseng_preds_best.csv"
    new_result = "./results/Mu-Ruei_Tseng_preds_{}.csv".format(args.model_name)

    best_df = pd.read_csv(best_result)
    new_df = pd.read_csv(new_result)

    acc = (best_df["label"] == new_df["label"]).mean()
    print("Accuracy: ", acc)

    # show where the model is wrong
    # visualize(args.data_path, output_path, args.model_name)
    # wrong_idx = np.where(best_df["label"] != new_df["label"])[0]
    # print("Number of wrong predictions: ", len(wrong_idx))
    # print("Wrong indices: ", wrong_idx)

    # debug_path = "./debug"
    # if os.path.exists(debug_path):
    #     shutil.rmtree(debug_path)
    # os.makedirs(debug_path)

    # test_image_path = os.path.join(args.data_path, "x_test_wr.npy")
    # test_images = np.load(test_image_path).reshape(-1, 28, 28)
    # for idx in wrong_idx:
    #     img = test_images[idx]
    #     plt.figure()
    #     plt.imshow(img, cmap="gray")
    #     plt.title(f"Prediction: {new_df['label'].iloc[idx]} - Truth: {best_df['label'].iloc[idx]}")
    #     plt.axis("off")
    #     plt.savefig(f"{debug_path}/{idx}.png")
    #     plt.close()
        