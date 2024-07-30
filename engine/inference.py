import torch
import torchvision.transforms as transforms

from PIL import Image

from models.cnn import CNN
from models.cnn_with_bn import CNNWithBN
from models.mlp import MLP


def load_image(img_path: str):
    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(dim=0)
    return img


def load_model(weight_path: str):
    model = CNN()
    model.load_state_dict(torch.load(weight_path))
    return model


@torch.no_grad()
def inference(weight_path: str, img_path: str):
    model = load_model(weight_path=weight_path)
    model.eval()
    img = load_image(img_path=img_path)
    output = model(img)
    result = torch.argmax(output, dim=1).item()
    return result
