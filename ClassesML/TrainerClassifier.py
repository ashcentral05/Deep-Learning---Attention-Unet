import torch
from Utilities.Utilities import Utilities


class TrainerUNET:
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter

    def set_model(self, model, device):
        self.model = model
        self.device = device

    def set_scope(self, scope):
        self.scope = scope

    def set_data(self, x_train, y_train, x_valid, y_valid):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def run(self):
        train_accuracy_dict = {}
        valid_accuracy_dict = {}
        train_accuracy_list = []
        valid_accuracy_list = []

        train_loss_list = []
        valid_loss_list = []

        for epoch in range(self.hyperparameter['max_epochs']):

            self.model.train()
            total_loss = 0.0
            total_accuracy = 0.0
            n_batch = len(self.x_train)
            for n in range(n_batch):
                x = self.x_train[n].to(self.device)
                y = self.y_train[n].to(self.device)

                y_hat = self.model(x)
                loss = self.scope.criterion(y_hat, y)

                self.scope.optimizer.zero_grad()
                loss.backward()
                self.scope.optimizer.step()
                total_loss += loss.item()

                preds = (y_hat > 0.5).float()
                #batch_accuracy = (preds == y).float().mean().item()
                intersection = (preds * y).sum(dim=(1, 2, 3))
                union = preds.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) - intersection
                batch_accuracy = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
                total_accuracy += batch_accuracy

            train_loss = total_loss / n_batch
            train_accuracy = total_accuracy / n_batch
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            print('epoch:' + str(epoch + 1) + '/' + str(self.hyperparameter['max_epochs']))
            print('train_loss:' + str(train_loss))
            print('train_accuracy:' + str(train_accuracy))

            self.model.eval()
            total_loss = 0.0
            total_accuracy = 0.0
            n_batch = len(self.x_valid)
            for n in range(n_batch):
                x = self.x_valid[n].to(self.device)
                y = self.y_valid[n].to(self.device)

                y_hat = self.model(x)
                loss = self.scope.criterion(y_hat, y)
                total_loss += loss.item()

                preds = (y_hat> 0.5).float()
                #batch_accuracy = (preds == y).float().mean().item()
                intersection = (preds * y).sum(dim=(1, 2, 3))
                union = preds.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) - intersection
                batch_accuracy = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
                total_accuracy += batch_accuracy

            valid_loss = total_loss / n_batch
            valid_accuracy = total_accuracy / n_batch
            valid_loss_list.append(valid_loss)
            valid_accuracy_list.append(valid_accuracy)
            print('epoch:' + str(epoch + 1) + '/' + str(self.hyperparameter['max_epochs']))
            print('valid_loss:' + str(valid_loss))
            print('valid_accuracy:' + str(valid_accuracy))

            if self.scope.scheduler is not None:
                validation_metric = valid_accuracy
                old_lr = self.scope.optimizer.param_groups[0]['lr']
                self.scope.scheduler.step(validation_metric)
                new_lr = self.scope.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f'Learning rate changed from {old_lr} to {new_lr} at epoch {epoch}')

            if self.scope.early_stopper:
                validation_metric = valid_accuracy
                self.model, keeptraining = self.scope.early_stopper.set(model=self.model, epoch=epoch,
                                                                        metric_epoch=validation_metric)
                if not keeptraining:
                    break

            train_accuracy_dict[epoch] = train_accuracy
            valid_accuracy_dict[epoch] = valid_accuracy

            train_accuracy_list = [train_accuracy_dict[e] for e in train_accuracy_dict.keys()]
            valid_accuracy_list = [valid_accuracy_dict[e] for e in valid_accuracy_dict.keys()]

        return train_accuracy_list, valid_accuracy_list,train_loss_list,valid_loss_list
