import matplotlib.pyplot as plt


def show_sample_images(images, labels, num_samples: int=16):
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(labels[i].item())
        plt.axis("off")
    plt.show()