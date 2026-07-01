import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from ClassesML.EarlyStopper import EarlyStopper
from Utilities.Utilities import Utilities

class ScopeClassifier:

    def __init__(self,model,hyperparameters):

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=hyperparameters['learning_rate'])

        if 'patience_lr' in hyperparameters:
            self.scheduler = ReduceLROnPlateau(self.optimizer,mode='max',
                                               patience=hyperparameters['patience_lr'],
                                               factor=0.1)
        else:
            self.scheduler = None

        if 'early_stopping' in hyperparameters:
            self.early_stopper = EarlyStopper(hyperparameters=hyperparameters)
        else:
            self.early_stopper = None

class ScopeGAN:
    def __init__(self,generator, discriminator,hyperparameters):
        self.criterion_generator = nn.MSELoss()
        self.criterion_discriminator = nn.BCELoss()
        self.optimizer_generator = optim.Adam(generator.parameters(),lr=hyperparameters['learning_rate'])
        self.optimizer_discriminator = optim.Adam(discriminator.parameters(),lr=hyperparameters['learning_rate'])

class ScopeAutoEncoder:
    def __init__(self,model,hyperparameters):
        self.criterion = nn.MSELoss()

        encoder_parameters = list(model.encoder.parameters())
        decoder_parameters = list(model.decoder.parameters())
        autoencoder_parameters = encoder_parameters + decoder_parameters

        self.optimizer = optim.Adam(autoencoder_parameters,lr=hyperparameters['learning_rate'])

class ScopeUNET:
    def __init__(self, model, hyperparameters,ioubool=False, dicebool=False):

        if ioubool:
            self.criterion = Utilities.IoULoss()
        elif dicebool:
            self.criterion = Utilities.DiceBCELoss()
        else:
            self.criterion = nn.MSELoss()
            
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=hyperparameters['learning_rate'])

        if 'patience_lr' in hyperparameters:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max',
                                               patience=hyperparameters['patience_lr'],
                                               factor=0.1)
        else:
            self.scheduler = None

        if 'early_stopping' in hyperparameters:
            self.early_stopper = EarlyStopper(hyperparameters=hyperparameters)
        else:
            self.early_stopper = None
