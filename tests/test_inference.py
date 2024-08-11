import pytest
import torch

from unittest.mock import MagicMock, patch

from engine import inference


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    model.load_state_dict = MagicMock()
    return model


@pytest.fixture
def mock_load_image():
    with patch("engine.inference.load_image") as mock:
        mock.return_value = torch.rand(1, 3, 28, 28)
        yield mock


@pytest.fixture
def mock_load_model(mock_model):
    with patch("engine.inference.load_model") as mock:
        mock.return_value = mock_model
        yield mock


@pytest.fixture
def mock_model_output():
    with patch("torch.argmax") as mock:
        mock.return_value.item.return_value = 1
        yield mock


@pytest.mark.parametrize("model_name", ["cnn", "cnn_with_bn", "mlp"])
def test_inference(model_name, mock_model, mock_load_image, mock_load_model, mock_model_output):
    device = torch.device("cpu")
    weight_path = "dummy_path/path/to/weights.pt"
    img_path = "dummy_path/path/to/image.jpg"

    result = inference(device, model_name, weight_path, img_path)

    # Veriffy that the model's load_state_dict method was called
    mock_load_model.assert_called_once_with(model_name=model_name, weight_path=weight_path)
    mock_model.load_state_dict.assert_called_once()

    # Verify that the load_image function was called
    mock_load_image.assert_called_once_with(img_path=img_path)

    # Verify that the model's to method was called
    mock_model.to.assert_called_once_with(device)

    # Verify that the model's eval method was called
    mock_model.eval.assert_called_once()

    # Verify that the output from the model's forward method is processed correctly
    mock_model_output.assert_called_once()
    assert result == 1
