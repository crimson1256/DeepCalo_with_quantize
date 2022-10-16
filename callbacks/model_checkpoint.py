import numpy as np
import json
from tensorflow.keras.callbacks import Callback


class ModelCheckpoint(Callback):
    """Save model weights and architecture separately.

    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
        multi_gpu_model: bool, whether a multi_gpu_model was used. Will ensure
            that only the core model's weights are saved.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', period=1,
                 multi_gpu_model=False):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.multi_gpu_model = multi_gpu_model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            filedir = filepath.rsplit('/',1)[0]
            filebasename = filepath.rsplit('/',1)[1]
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filedir))
                        self.best = current

                        # Save weights
                        if self.multi_gpu_model:
                            self.model.layers[-2].save_weights(filedir + 'weights.' + filebasename + '.hdf5', overwrite=True)
                        else:
                            self.model.save_weights(filedir + 'weights.' + filebasename + '.hdf5', overwrite=True)

                        # Save model to JSON
                        if self.multi_gpu_model:
                            model_json = self.model.layers[-2].to_json()
                        else:
                            model_json = self.model.to_json()
                        with open(filedir + 'model.' + filebasename + '.json', "w") as json_file:
                            json.dump(model_json, json_file)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filedir))

                # Save weights
                if self.multi_gpu_model:
                    self.model.layers[-2].save_weights(filedir + 'weights.' + filebasename + '.hdf5', overwrite=True)
                else:
                    self.model.save_weights(filedir + 'weights.' + filebasename + '.hdf5', overwrite=True)

                # Save model to JSON
                if self.multi_gpu_model:
                    model_json = self.model.layers[-2].to_json()
                else:
                    model_json = self.model.to_json()
                with open(filedir + 'model.' + filebasename + '.json', "w") as json_file:
                    json.dump(model_json, json_file)
