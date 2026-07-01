import torch
import matplotlib.pyplot as plt


def visualize_predictions(model, x, y, device, n_samples=4):

    model.eval()
    with torch.no_grad():
        x_sample = x[:n_samples].to(device)
        y_sample = y[:n_samples].to(device)
        y_hat = model(x_sample)
        preds = (y_hat > 0.5).float()

    x_sample = x_sample.cpu()
    y_sample = y_sample.cpu()
    preds = preds.cpu()

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))

    for i in range(n_samples):
        axes[i, 0].imshow(x_sample[i, 0], cmap='gray')
        axes[i, 0].set_title('image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(y_sample[i, 0], cmap='gray')
        axes[i, 1].set_title('ground truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(preds[i, 0], cmap='gray')
        axes[i, 2].set_title('prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()