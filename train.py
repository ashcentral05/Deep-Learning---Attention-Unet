import os
import time
import matplotlib
matplotlib.use('Agg')
import torch
from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.UNET_NoAttention import UNET_NoAttention
from ClassesML.UNET import UNET
from ClassesML.Scope import ScopeUNET
from ClassesML.TrainerUNET import TrainerUNET
from Utilities.Utilities import Utilities

print(time.ctime())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_parent_project = os.getcwd()
dataset_image_path = os.path.join(path_parent_project, "Dataset", "UNET")
dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

hyperparameters = dict(input_dim=1,
                        output_dim=1,
                        filters=[32, 64, 128, 256],
                        kernel_size=2,
                        embedding_dim=None,
                        d_model=None,
                        activation='relu',
                        batch_normalization=True,
                        dropout_rate=0.05,
                        learning_rate=0.0001,
                        early_stopping=True,
                        patience_lr=5,
                        max_epochs=30)

model = UNET(hyperparameters).to(device)
scope = ScopeUNET(model, hyperparameters)

x_train = train_dataset[0]
y_train = train_dataset[1]
x_val = val_dataset[0]
y_val = val_dataset[1]

trainer = TrainerUNET(hyperparameter=hyperparameters)
trainer.set_model(model=model, device=device)
trainer.set_scope(scope=scope)
trainer.set_data(x_train=x_train, y_train=y_train,
                  x_valid=x_val, y_valid=y_val)
train_accuracy_list, valid_accuracy_list, train_loss_list, valid_loss_list = trainer.run()
print(time.ctime())

run_id = time.strftime("%Y%m%d_%H%M%S")

model_dir = os.path.join(path_parent_project, "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "unet.pth")
torch.save({'model_state_dict': model.state_dict(),
            'hyperparameters': hyperparameters,
            'run_id': run_id}, model_path)
print('model saved to ' + model_path)

result_dir = os.path.join(path_parent_project, "Result", run_id)
Utilities.plot_curves(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, result_dir=result_dir)
