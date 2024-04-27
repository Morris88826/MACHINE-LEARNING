import os
import torch
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from libs.dataloader import Loader
from config import get_model
from matplotlib_venn import venn3

def init_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join('./checkpoints', f'{model_name}.pth')
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Model checkpoint not found at {ckpt_path}")
    
    model = get_model(model_name)
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    print(f"Load model from {ckpt_path} successfully")
    return model

def create_venn_diagram(error_pred_idx, output_path):

    intersection = set(error_pred_idx["image"]).intersection(set(error_pred_idx["audio"]), set(error_pred_idx["hybrid"]))
    image_exclusive = set(error_pred_idx["image"]) - set(error_pred_idx["audio"]) - set(error_pred_idx["hybrid"])
    audio_exclusive = set(error_pred_idx["audio"]) - set(error_pred_idx["image"]) - set(error_pred_idx["hybrid"])
    hybrid_exclusive = set(error_pred_idx["hybrid"]) - set(error_pred_idx["image"]) - set(error_pred_idx["audio"])
    image_audio = set(error_pred_idx["image"]).intersection(set(error_pred_idx["audio"])) - set(error_pred_idx["hybrid"])
    image_hybrid = set(error_pred_idx["image"]).intersection(set(error_pred_idx["hybrid"])) - set(error_pred_idx["audio"])
    audio_hybrid = set(error_pred_idx["audio"]).intersection(set(error_pred_idx["hybrid"])) - set(error_pred_idx["image"])

    print("Intersection:", len(intersection))
    print("Image Exclusive:", len(image_exclusive))
    print("Audio Exclusive:", len(audio_exclusive))
    print("Hybrid Exclusive:", len(hybrid_exclusive))
    print("Image & Audio:", len(image_audio))
    print("Image & Hybrid:", len(image_hybrid))
    print("Audio & Hybrid:", len(audio_hybrid))


    plt.figure(figsize=(10, 5))
    plt.title("Venn Diagram of Error Indices", fontsize=20)

    venn3(subsets=(len(image_exclusive), len(audio_exclusive), len(image_audio), len(hybrid_exclusive), len(image_hybrid), len(audio_hybrid), len(intersection)), set_labels=("ImageNet_CNN", "AudioNet_CNN", "HybridNet_CNN"))
    plt.savefig(f"{output_path}/venn_diagram.png")
    plt.close()

def main(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = init_model("ImageNet_CNN")
    audio_model = init_model("AudioNet_CNN")
    hybrid_model = init_model("HybridNet_CNN")
    

    loader = Loader(data_path, batch_size=32)

    error_images = []
    predictions = {
        "image": [],
        "audio": [],
        "hybrid": []
    }
    error_pred_idx = {
        "image": [],
        "audio": [],
        "hybrid": []
    }
    error_gt = []
    index = 0
    with torch.no_grad():
        for _, (image, audio, label) in enumerate(loader.test_loader):
            image = Variable(image).float().to(device)
            audio = Variable(audio).float().to(device)
            label = Variable(label).to(device)

            
            output1 = image_model(image)
            output2 = audio_model(audio)
            output3 = hybrid_model(image, audio)

            _, pred1 = torch.max(output1, 1)
            _, pred2 = torch.max(output2, 1)
            _, pred3 = torch.max(output3, 1)

            image = image.cpu().numpy()
            pred1 = pred1.cpu().numpy()
            pred2 = pred2.cpu().numpy()
            pred3 = pred3.cpu().numpy()
            label = label.cpu().numpy()

            wrong_idx = np.where(pred1 != label)[0]
            error_images.extend(image[wrong_idx])
            predictions["image"].extend(pred1[wrong_idx])
            predictions["audio"].extend(pred2[wrong_idx])
            predictions["hybrid"].extend(pred3[wrong_idx])

            error_gt.extend(label[wrong_idx])

            error_pred_idx["image"].extend(np.where(pred1 != label)[0] + index)
            error_pred_idx["audio"].extend(np.where(pred2 != label)[0] + index)
            error_pred_idx["hybrid"].extend(np.where(pred3 != label)[0] + index)

            index += label.shape[0]



        output_path = "./demo"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        create_venn_diagram(error_pred_idx, output_path)

        # randomly sample 10 error images
        np.random.seed(0)
        error_idx = np.random.choice(len(error_images), 10, replace=False)


        fig, ax = plt.subplots(2, 5, figsize=(20, 10), dpi=100)
        for i, idx in enumerate(error_idx):
            error_image = error_images[idx].reshape(28, 28)
            
            ax[i//5, i%5].imshow(error_image, cmap="gray")
            ax[i//5, i%5].set_title(f"Prediction: ({predictions['image'][idx]},{predictions['audio'][idx]},{predictions['hybrid'][idx]}), GT: {error_gt[idx]}", fontsize=15)
            ax[i//5, i%5].axis("off")
        
        plt.suptitle(f"Predictions - (ImageNet_CNN, AudioNet_CNN, HybridNet_CNN)", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{output_path}/comparisons.png")
        plt.close()

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on MNIST dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data directory")
    args = parser.parse_args()

    main(args.data_path)
    