from models import CNN, CNNWithBN, MLP


class Exporter(object):
    def __init__(self, model_name: str, weight_dir: str):
        self.model_name = model_name
        self.weight_dir = weight_dir

        assert model_name in ("cnn", "cnn_with_bn", "mlp")

        if model_name == "cnn":
            self.model = CNN()
        elif model_name == "cnn_with_bn":
            self.model = CNNWithBN()
        else:
            self.model = MLP()

    def __call__(self):
        exported_model = self.export()
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
