import os
import numpy as np
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
from torchvision.utils import make_grid

class Utilities:

    @staticmethod
    def get_activation(activation_str):

        if activation_str == 'relu':
            return nn.ReLU()
        elif activation_str == 'sigmoid':
            return nn.Sigmoid()
        elif activation_str == 'tanh':
            return nn.Tanh()
        elif activation_str == "linear":
            return None
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")

    @staticmethod
    def compute_accuracy(y_hat, y):

        if not isinstance(y_hat, torch.Tensor):
            y_hat = torch.tensor(y_hat)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        _, predicted = torch.max(y_hat, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0) * 100

        return accuracy

    @staticmethod
    def DiceBCELoss(y_hat, y):
        bce = nn.BCELoss()
        bce_loss = bce(y_hat, y)

        intersection = (y_hat * y).sum(dim=(1, 2, 3))
        dice_loss = 1 - (
                (2 * intersection + 1e-6) /
                (y_hat.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1e-6)
        )

        return bce_loss + dice_loss.mean()

    @staticmethod
    def plot_curves(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, result_dir):
        os.makedirs(result_dir, exist_ok=True)

        plt.figure()
        plt.plot(train_loss_list, label='train')
        plt.plot(valid_loss_list, label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'loss.png'))
        plt.show()

        plt.figure()
        plt.plot(train_accuracy_list, label='train')
        plt.plot(valid_accuracy_list, label='val')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'accuracy.png'))
        plt.show()

    @staticmethod
    def visualize_predictions(model, x, y, device, n_samples=4, save_dir='.'):
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
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'visualize.png'))
        plt.show()

    @staticmethod
    def IoULoss(y_hat, y, eps=1e-6):
        y_hat = y_hat.view(y_hat.size(0), -1)
        y = y.view(y.size(0), -1)

        intersection = (y_hat * y).sum(dim=1)
        union = y_hat.sum(dim=1) + y.sum(dim=1) - intersection

        iou = (intersection + eps) / (union + eps)

        return 1 - iou.mean()
