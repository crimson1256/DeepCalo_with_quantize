import numpy as np
import h5py
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
plt.switch_backend('agg') # For outputting plots when running on a server
plt.ioff() # Turn off interactive mode, so that figures are only saved, not displayed
plt.rcParams.update({'font.size': 18})

class LRFinder(Callback):

    '''
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss_vs_lr()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        scale: Whether to scan learning rate range linearly (`'linear'`) or logarithmically (`'log'`).
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e-2, scale='linear',
                 steps_per_epoch=None, epochs=None, fig_dir='./'):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.scale = scale
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        self.fig_dir = fig_dir

        if self.scale == 'log':
            self.log_range = np.logspace(np.log10(self.min_lr), np.log10(self.max_lr),
                                         np.ceil(self.total_iterations+1))

    def clr(self):
        '''Calculate the learning rate.'''
        if self.scale == 'linear':
            x = self.iteration / self.total_iterations
            return self.min_lr + (self.max_lr-self.min_lr) * x
        elif self.scale == 'log':
            return self.log_range[self.iteration]

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        if hasattr(self.model.optimizer, 'lr'):
            K.set_value(self.model.optimizer.lr, self.min_lr)
        else:
            K.set_value(self.model.optimizer.optimizer._lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        if hasattr(self.model.optimizer, 'lr'):
            self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        else:
            self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.optimizer._lr))

        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if hasattr(self.model.optimizer, 'lr'):
            K.set_value(self.model.optimizer.lr, self.clr())
        else:
            K.set_value(self.model.optimizer.optimizer._lr, self.clr())

    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''

        plt.plot(self.history['iterations'], self.history['lr'])
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        # plt.yscale('log')
        plt.tight_layout()
        plt.savefig(self.fig_dir + 'lr.pdf')
        plt.close()

    def plot_loss_vs_lr(self, chosen_limits=None):
        '''Helper function to quickly observe the learning rate experiment results.'''

        mask = np.array(self.history['loss']) < 1e3
        lrs = np.array(self.history['lr'])[mask]
        losses = np.array(self.history['loss'])[mask]

        fig,ax = plt.subplots()
        ax.plot(lrs, losses)
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')
        if chosen_limits is not None:
            ax.axvline(x=chosen_limits[0], color='k', ls=':', label='Chosen limits')
            ax.axvline(x=chosen_limits[1], color='k', ls=':')
            ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(self.fig_dir + 'loss_vs_lr.pdf')
        plt.close()

        with h5py.File(self.fig_dir + 'lr_vs_loss.h5', 'w') as h:
            h.create_dataset('lr', data=lrs)
            h.create_dataset('loss', data=losses)
