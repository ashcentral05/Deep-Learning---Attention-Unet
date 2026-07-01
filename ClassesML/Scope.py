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
        self, model, hyperparameters, ioubool=False, dicebool=False, weight_decay=0.0
    ):

        if ioubool:
            self.criterion = Utilities.IoULoss
        elif dicebool:
            self.criterion = Utilities.DiceBCELoss
        else:
            self.criterion = nn.MSELoss()

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
