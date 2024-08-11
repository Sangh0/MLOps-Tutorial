import pytest
import torch

from unittest.mock import MagicMock

from engine import evaluate


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.eval = MagicMock()
    return model


@pytest.fixture
def mock_test_loader():
    class MockLoader:
        def __iter__(self):
            for _ in range(3): # simulate 3 batches
                images = torch.rand(2, 3, 28, 28)
                labels = torch.randint(0, 10, (2,))
                yield images, labels

    return MockLoader()


@pytest.fixture
def mock_cal_accuracy():
    with pytest.MonkeyPatch().context() as m:
        # Mock the cal_accuracy function
        mock_cal_acc = MagicMock(return_value=torch.tensor(0.9))
        m.setattr("utils.cal_accuracy", mock_cal_acc)
        yield mock_cal_acc


def test_evaluate(mock_model, mock_test_loader, mock_cal_accuracy):
    device = torch.device("cpu")
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = None

    # Run the evaluate function
    evaluate(device, mock_model, mock_test_loader)
    
    # Verify the model's eval method was called
    mock_model.eval.assert_called_once()

    # Verify the cal_accuracy function was called
    assert mock_cal_accuracy.call_count == 3 
