from libs.models.image import ImageNet_CNN
from libs.models.audio import AudioNet_LSTM, AudioNet_CNN
from libs.models.hybrid import HybridNet_CNN, HybridNet_CNN_LSTM

model_names = ["ImageNet_CNN", "AudioNet_CNN", "AudioNet_LSTM", "HybridNet_CNN", "HybridNet_CNN_LSTM"]

def choose_model():
    title = ['{}: for {}'.format(i, name) for i, name in enumerate(model_names)]
    model = int(input("Choose the model ({}): ".format(", ".join(title))))
    for i in range(len(model_names)):
        if model == i:
            return model_names[i]
    raise ValueError("Invalid model choice")

def get_model(name):
    if name == "ImageNet_CNN":
        return ImageNet_CNN()
    elif name == "AudioNet_CNN":
        return AudioNet_CNN()
    elif name == "AudioNet_LSTM":
        return AudioNet_LSTM(embedding_size=13, bidirectional=False)
    elif name == "HybridNet_CNN":
        return HybridNet_CNN()
    elif name == "HybridNet_CNN_LSTM":
        return HybridNet_CNN_LSTM(embedding_size=13)
    else:
        raise ValueError(f"Invalid model name: {name}")

def get_hyperparameters(name):
    
    if name == "ImageNet_CNN":
        return {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
        }
    elif name == "AudioNet_CNN":
        return {
            "lr": 1e-4,
            "epochs": 100,
            "batch_size": 64,
        }
    elif name == "AudioNet_LSTM":
        return {
            "lr": 5e-4,
            "epochs": 50,
            "batch_size": 64,
        }
    elif name == "HybridNet_CNN":
        return {
            "lr": 0.001,
            "epochs": 20,
            "batch_size": 32,
        }
    elif name == "HybridNet_CNN_LSTM":
        return {
            "lr": 1e-4,
            "epochs": 30,
            "batch_size": 64,
        }
    else:
        raise ValueError(f"Invalid model name: {name}")
