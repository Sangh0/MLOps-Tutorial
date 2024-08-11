import torch

from models import CNN, CNNWithBN, MLP


class Exporter(object):
    def __init__(self, model_name: str, weight_dir: str, save_dir: str):
        self.model_name = model_name
        self.weight_dir = weight_dir
        self.save_dir = save_dir

        assert model_name in ("cnn", "cnn_with_bn", "mlp")

        if model_name == "cnn":
            self.model = CNN()
        elif model_name == "cnn_with_bn":
            self.model = CNNWithBN()
        else:
            self.model = MLP()

        self.model.load_state_dict(torch.load(weight_dir))

    def __call__(self):
        quantized_model = self.export()

        dummy_input = torch.randn(1, 1, 28, 28)
        quantized_model(dummy_input)
        quantized_model.save(self.save_dir)
        print(f"Quantized model saved to {self.save_dir}")
        return quantized_model

    def export(self, backend="x86"):
        self.model.eval()
        self.model = self.model.to("cpu")
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(self.model, inplace=True)
        torch.quantization.convert(self.model, inplace=True)
        return self.model
