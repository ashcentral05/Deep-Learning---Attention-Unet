import os
import numpy as np
import matplotlib
from torch import cpu

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.UNET import UNET
from ClassesML.Scope import ScopeUNET
from ClassesML.TrainerClassifier import TrainerUNET
import torch
import torch.optim as optim
from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_parent_project = os.getcwd()
dataset_image_path = path_parent_project + "\\Dataset\\"+"\\UNET\\"
dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

hyperparameters = dict(input_dim = 1,
                        output_dim = 1,
                        filters = [16, 32, 64, 128],
                        kernel_size = 2,
                        embedding_dim = None,
                        d_model = None,
                        activation = 'relu',
                        batch_normalization = True,
                        dropout_rate = 0.01,
                        learning_rate = 0.0001,
                        early_stopping = True,
                        max_epochs = 50)

model = UNET(hyperparameters).to(device)
print(model)
scope = ScopeUNET(model,hyperparameters)

input_size = (4, input_dim[0], input_dim[1], input_dim[2])
input_data = torch.rand(input_size, device=device)
print(summary(model=model, input_data = input_data,depth=0))

x_train = train_dataset[0]
y_train = train_dataset[1]
x_val = val_dataset[0]
y_val = val_dataset[1]

trainer = TrainerUNET(hyperparameter = hyperparameters)
trainer.set_model(model=model,device=device)
trainer.set_scope(scope=scope)
trainer.set_data(x_train=x_train,y_train=y_train,
                 x_valid=x_val,y_valid=y_val,)
train_accuracy_list, valid_accuracy_list = trainer.run()
