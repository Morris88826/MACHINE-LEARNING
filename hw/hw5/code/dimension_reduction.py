import os
import torch
import argparse
import numpy as np
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.autograd import Variable
from libs.dataloader import Loader
from config import get_model, choose_model
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def cal_dimension_reduction(features):
    # use pca first to reduce the dimensionality
    pca = PCA(n_components=50)
    features = pca.fit_transform(features)

    # # use t-sne to reduce the dimensionality
    tsne = TSNE(n_components=2, random_state=42)  # Set random_state for reproducibility
    features = tsne.fit_transform(features)

    return features

def plot_embeddings(em1, labels, model_name, em2=None, out_dir="./demo"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    def plot_scatter(ax, embeddings, labels, title):    
        ax.set_title(title)
        colors = matplotlib.colormaps.get_cmap('tab10')
        for i in range(10):
            indices = labels == i
            ax.scatter(embeddings[indices, 0], embeddings[indices, 1], label=str(i), color=colors(i), s=5)
        ax.grid(True)
        ax.legend(markerscale=2)

    def plot_kmeans(ax, embeddings, title):
        kmeans = KMeans(n_clusters=10, n_init=10, random_state=42).fit(embeddings)
        ax.set_title(title)
        colors = matplotlib.colormaps.get_cmap('tab10')
        for i in range(10):
            indices = kmeans.labels_ == i
            ax.scatter(embeddings[indices, 0], embeddings[indices, 1], label=str(i), color=colors(i), s=5)
            centroid = kmeans.cluster_centers_[i]
            ax.scatter(centroid[0], centroid[1], s=100, marker='x', color='black')
        ax.grid(True)

    if em2 is not None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        plot_scatter(ax[0], em1, labels, f"{model_name} Image Embeddings")
        plot_scatter(ax[1], em2, labels, f"{model_name} Audio Embeddings")
        plt.savefig(f"{out_dir}/{model_name}_emb.png")
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        plot_kmeans(ax[0], em1, f"{model_name} Image Embeddings with KMeans")
        plot_kmeans(ax[1], em2, f"{model_name} Audio Embeddings with KMeans")
        plt.savefig(f"{out_dir}/{model_name}_emb_kmeans.png")
        plt.close()
    else:
        plt.figure(figsize=(8, 8))
        plot_scatter(plt.gca(), em1, labels, f"{model_name} Embeddings")
        plt.savefig(f"{out_dir}/{model_name}_emb.png")
        plt.close()

        plt.figure(figsize=(8, 8))
        plot_kmeans(plt.gca(), em1, f"{model_name} Embeddings with KMeans")
        plt.savefig(f"{out_dir}/{model_name}_emb_kmeans.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model on MNIST dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--model_name", "-m", type=str, help="Type of model to use")
    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = choose_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = Loader(args.data_path, batch_size=32)
    model = get_model(args.model_name)
    model.to(device)

    ckpt_path = os.path.join('./checkpoints', f'{args.model_name}.pth')
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Model checkpoint not found at {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    image_embs = []
    audio_embs = []
    labels = []
    print("Evaluating model: {}...".format(args.model_name))
    for (image, audio, label) in tqdm.tqdm(loader.train_loader):
        image = Variable(image).float().to(device)
        audio = Variable(audio).float().to(device)
        label = Variable(label).to(device)

        image_emb = None
        audio_emb = None
        if "Image" in args.model_name:
            image_emb = model(image, embedding=True)
            image_emb = image_emb.cpu().detach().numpy()
        elif "Audio" in args.model_name:
            audio_emb = model(audio, embedding=True)
            audio_emb = audio_emb.cpu().detach().numpy()
        elif "Hybrid" in args.model_name:
            image_emb, audio_emb = model(image, audio, embedding=True)
            image_emb = image_emb.cpu().detach().numpy()
            audio_emb = audio_emb.cpu().detach().numpy()
        else:
            raise ValueError(f"Invalid model name: {args.model_name}")
        
        labels.append(label.cpu().detach().numpy())

        if image_emb is not None:
            image_embs.append(image_emb)
        if audio_emb is not None:
            audio_embs.append(audio_emb)

    labels = np.concatenate(labels, axis=0)

    if len(image_embs) > 0:
        image_embs = np.concatenate(image_embs, axis=0)
        print("Image embeddings shape:", image_embs.shape)
        image_embs_dr = cal_dimension_reduction(image_embs)

    if len(audio_embs) > 0:
        audio_embs = np.concatenate(audio_embs, axis=0)
        print("Audio embeddings shape:", audio_embs.shape)
        audio_embs_dr = cal_dimension_reduction(audio_embs)
    
    if "Image" in args.model_name:
        plot_embeddings(image_embs_dr, labels, args.model_name)
    elif "Audio" in args.model_name:
        plot_embeddings(audio_embs_dr, labels, args.model_name)
    elif "Hybrid" in args.model_name:
        plot_embeddings(image_embs_dr, labels, args.model_name, em2=audio_embs_dr)
    
