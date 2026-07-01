import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from ClassesML.EarlyStopper import EarlyStopper
from Utilities.Utilities import Utilities


class ScopeClassifier:
    def __init__(self, model, hyperparameters):

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=hyperparameters["learning_rate"]
        )

        if "patience_lr" in hyperparameters:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=hyperparameters["patience_lr"],
                factor=0.1,
            )
        else:
            self.scheduler = None

        if "early_stopping" in hyperparameters:
            self.early_stopper = EarlyStopper(hyperparameters=hyperparameters)
        else:
            self.early_stopper = None


class ScopeUNET:
    def __init__(
        self,
        model,
        hyperparameters,
        loss = "bce",
        weight_decay=0.0,
    ):

        if loss == "iou":
            self.criterion = Utilities.IoULoss
        elif loss == "dice":
            self.criterion = Utilities.DiceBCELoss
        elif loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "bce":
            self.criterion = nn.BCELoss()

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=weight_decay,
        )

        if "patience_lr" in hyperparameters:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=hyperparameters["patience_lr"],
                factor=0.1,
            )
        else:
            self.scheduler = None

        if "early_stopping" in hyperparameters:
            self.early_stopper = EarlyStopper(hyperparameters=hyperparameters)
        else:
            self.early_stopper = None
