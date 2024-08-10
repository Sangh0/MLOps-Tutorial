import torch
import torchvision.transforms as transforms

from PIL import Image

from models import CNN, CNNWithBN, MLP


def load_image(img_path: str):
    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(dim=0)
    return img


def load_model(model_name: str, weight_path: str):
    assert model_name in ("cnn", "cnn_with_bn", "mlp")

    if model_name == "cnn":
        model = CNN()
    elif model_name == "cnn_with_bn":
        model = CNNWithBN()
    else:
        model = MLP()

    model.load_state_dict(torch.load(weight_path))
    return model


@torch.no_grad()
def inference(
    device: torch.device,
    model_name: str,
    weight_path: str,
    img_path: str,
):
    model = load_model(model_name=model_name, weight_path=weight_path)
    model = model.to(device)
    model.eval()
    img = load_image(img_path=img_path).to(device)
    output = model(img)
    result = torch.argmax(output, dim=1).item()
    return result
