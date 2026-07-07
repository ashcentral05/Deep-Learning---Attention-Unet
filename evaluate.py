import os
import matplotlib
matplotlib.use('Agg')
import torch
from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.UNET import UNET
from Utilities.Utilities import Utilities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_parent_project = os.getcwd()
dataset_image_path = path_parent_project + "\\Dataset\\" + "\\UNET\\"
dataset = DatasetLoader(root=dataset_image_path)
train_dataset, val_dataset, input_dim, n_classes = dataset.load_images_labels_data()

x_val = val_dataset[0]
y_val = val_dataset[1]

model_path = os.path.join(path_parent_project, "model", "unet.pth")
if not os.path.exists(model_path):
    raise ValueError('no trained model found at ' + model_path + ', please execute train.py first.')

checkpoint = torch.load(model_path, map_location=device)
hyperparameters = checkpoint['hyperparameters']
run_id = checkpoint['run_id']
result_dir = os.path.join(path_parent_project, "Result", run_id)

model = UNET(hyperparameters).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

total_accuracy = 0.0
n_batch = len(x_val)
with torch.no_grad():
    for n in range(n_batch):
        x = x_val[n].to(device)
        y = y_val[n].to(device)
        y_hat = model(x)
        preds = (y_hat > 0.5).float()
        batch_accuracy = (preds == y).float().mean().item()
        total_accuracy += batch_accuracy

valid_accuracy = total_accuracy / n_batch
print('valid_accuracy:' + str(valid_accuracy))

Utilities.visualize_predictions(model, x_val[0], y_val[0], device, n_samples=2, save_dir=result_dir)
