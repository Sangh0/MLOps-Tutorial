import torch

from app.utils.utils import cal_accuracy


@torch.no_grad()
def evaluate(device, model, test_loader):
    model.eval()
    test_acc = 0
    for batch, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        acc = cal_accuracy(outputs, labels)
        test_acc += acc.item()

    print(f"Test Accuracy: {test_acc/(batch+1):.3f}")
