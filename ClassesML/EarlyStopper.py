import numpy as np

class EarlyStopper:

    def __init__(self,hyperparameters):

        self.hyperparameters = hyperparameters
        self.max_epoch = self.hyperparameters['max_epochs']
        self.metric_epochs = []
        self.early_stopping_patience = 20
        self.best_model_weights = None

    def set(self, model ,epoch, metric_epoch):
        keep_training = True
        self.metric_epochs.append(metric_epoch)

    #keep the weight at the beginning
        if epoch ==0:
            self.wait = 0
            print('Epoch'+str(epoch)+' - keeping weights')
            self._keep_weights(model)
    #check if metric
        elif metric_epoch > np.max(self.metric_epochs[:-1]):
            self.wait = 0
            print('Epoch'+str(epoch)+' - best metrics:'
            +str(metric_epoch) +' - Keeping weights')
            self._keep_weights(model)
        else:
            self.wait +=1
            print('Epoch'+str(epoch)+
                  ' - Metrics did not improve, wait'+str(self.wait))
            if self.wait >= self.early_stopping_patience:
                print('Epoch'+str(epoch)+
                      'Patience reached - Stop training at epoch '
                      +str(epoch)+' - Restoring weights')
                keep_training = False
                self._restore_weights(model)
        if epoch == (self.max_epoch - 1):
            print('Max epoch reached - Stop training - Restoring weights')
            keep_training = False
            self._restore_weights(model)
        return model, keep_training

    def _restore_weights(self, model):
        model.load_state_dict(self.best_model_weights)
    def _keep_weights(self, model):
        self.best_model_weights = model.state_dict().copy()