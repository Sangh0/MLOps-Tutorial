import pytest
import mlflow
import torch
import torch.nn as nn

from unittest.mock import MagicMock, patch

from engine.train import Trainer


@pytest.fixture
def mock_model():
    model = MagicMock(spec=nn.Module)
    model.to = MagicMock(return_value=model)
    model.train = MagicMock()
    model.eval = MagicMock()
    model.load_state_dict = MagicMock()
    return model


@pytest.fixture
def mock_optimizer():
    optimizer = MagicMock()
    return optimizer


@pytest.fixture
def mock_mlflow_run():
    return MagicMock(spec=mlflow.ActiveRun)


@pytest.fixture
def mock_train_loader():
    return MagicMock()


@pytest.fixture
def mock_valid_loader():
    return MagicMock()


@pytest.fixture
def trainer(mock_model, mock_optimizer, mock_mlflow_run):
    # Mock mlflow and os
    with patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.pytorch.log_model') as mock_log_model, \
         patch('mlflow.register_model') as mock_register_model, \
         patch('mlflow.tracking.MlflowClient') as mock_mlflow_client, \
         patch('os.makedirs') as mock_makedirs:
        mlflow_client_instance = MagicMock()
        mock_mlflow_client.return_value = mlflow_client_instance
        mlflow_client_instance.get_latest_versions.return_value = [MagicMock(version='1')]
        
        return Trainer(
            device='cpu',
            model=mock_model,
            lr=0.01,
            weight_decay=0.005,
            epochs=1,
            optimizer_name='sgd',
            exp_name='test_exp',
            mlflow_run=mock_mlflow_run,
        )
    

def test_train_on_batch(trainer, mock_train_loader):
    # Prepare mock outputs
    trainer.model = MagicMock()
    trainer.model.forward = MagicMock(return_value=(torch.rand(10, 3),))
    trainer.loss_func = MagicMock()
    trainer.loss_func.return_value = MagicMock()
    trainer.loss_func.return_value.item.return_value = 0.1
    
    # Mock accuracy
    with patch('utils.utils.cal_accuracy') as mock_cal_accuracy:
        mock_cal_accuracy.return_value = torch.tensor(0.9)
        loss, acc = trainer.train_on_batch(trainer.model, mock_train_loader)
        assert loss >= 0
        assert acc >= 0


def test_valid_on_batch(trainer, mock_valid_loader):
    # Prepare mock outputs
    trainer.model = MagicMock()
    trainer.model.forward = MagicMock(return_value=(torch.rand(10, 3),))
    trainer.loss_func = MagicMock()
    trainer.loss_func.return_value = MagicMock()
    trainer.loss_func.return_value.item.return_value = 0.1

    # Mock accuracy
    with patch('utils.utils.cal_accuracy') as mock_cal_accuracy:
        mock_cal_accuracy.return_value = torch.tensor(0.9)
        loss, acc = trainer.valid_on_batch(trainer.model, mock_valid_loader)
        assert loss >= 0
        assert acc >= 0


@patch('engine.train.Trainer.train_on_batch')
@patch('engine.train.Trainer.valid_on_batch')
@patch('torch.save')
def test_fit(mock_save, mock_valid_on_batch, mock_train_on_batch, trainer, mock_train_loader, mock_valid_loader):
    # Mock methods
    mock_train_on_batch.return_value = (0.1, 0.9)
    mock_valid_on_batch.return_value = (0.1, 0.9)

    trainer.fit(mock_train_loader, mock_valid_loader)

    # Verify methods are called
    mock_train_on_batch.assert_called_once()
    mock_valid_on_batch.assert_called_once()
    mock_save.assert_called_once()

    # Verify MLFlow logging
    with patch('mlflow.log_metric') as mock_log_metric:
        mock_log_metric.assert_any_call("train_loss", 0.1, step=0)
        mock_log_metric.assert_any_call("train_acc", 0.9, step=0)
        mock_log_metric.assert_any_call("valid_loss", 0.1, step=0)
        mock_log_metric.assert_any_call("valid_acc", 0.9, step=0)
        mock_log_metric.assert_any_call("lr", 0.01)

    # Verify that the model is logged and registered correctly
    with patch('mlflow.pytorch.log_model') as mock_log_model, \
         patch('mlflow.register_model') as mock_register_model:
        mock_log_model.assert_called_once()
        mock_register_model.assert_called_once()
        assert mock_register_model.call_args[0][0] == f"runs/{trainer.mlflow_run.info.run_id}/model"
