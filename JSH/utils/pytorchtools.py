import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0.015, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False

        self.val_loss_min = np.Inf
        self.val_acc_max = 0
        self.saved= False
        
        

    def __call__(self, val_loss, val_acc,  model):

        score = -val_loss

        if self.best_val_loss is None:
            self.best_val_loss = score
            self.saved = True
            self.save_checkpoint(val_loss, val_acc, model)

        elif score <= self.best_val_loss + self.delta: # If loss shows no improvement in performance more than delta value,  add counter
            self.counter += 1
            self.saved = False

            print()
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.counter = 0
            self.best_val_loss = score
            self.saved = True
            self.save_checkpoint(val_loss, val_acc, model)

    def save_checkpoint(self, val_loss, val_acc, model):
        '''Saves model when validation loss decrease within delta value'''
        print("\n\nNew best model Created! Saving the best model..")

        torch.save(model.module.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.val_acc_max  = val_acc