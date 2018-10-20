# -*- coding: utf-8 -*-

import logging
import os
import time
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop, SGD

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, model, train_stream, dev_stream, callbacks,
                 initial_epoch, num_epochs, serialization_dir):
        self.model = model
        self.train_stream = train_stream
        self.dev_stream = dev_stream
        self.serialization_dir = serialization_dir

        self._callbacks = callbacks
        self._initial_epoch = initial_epoch
        self._num_epochs = num_epochs

    @staticmethod
    def compile_model(model, config):
        if config['optimizer'] == 'rmsprop':
            optimizer_fn = RMSprop
        elif config['optimizer'] == 'adam':
            optimizer_fn = Adam
        elif config['optimizer'] == 'sgd':
            optimizer_fn = SGD
        else:
            raise ValueError("Unknown optimizer")

        model.compile(optimizer=optimizer_fn(lr=config['learning_rate'],
                                             clipnorm=config['clip_norm']),
                      loss=config['loss'],
                      metrics=config['metrics'])

        return model

    @classmethod
    def from_params(cls, config, model, train_stream, dev_stream, serialization_dir,
                    continue_training=True):
        callbacks = []

        def time_callback(epoch, logs):
            t = getattr(time_callback, "t")
            logs['time_epoch'] = time.time() - t
            setattr(time_callback, "t", time.time())

        def print_callback(epoch, logs):
            for k, v in list(logs.items()):
                logger.info("{}={}".format(k, v))

        def history_callback(epoch, logs):
            history_path = os.path.join(serialization_dir, "history.csv")
            if os.path.exists(history_path):
                H = pd.read_csv(history_path)
                H = {col: list(H[col].values) for col in H.columns}
            else:
                H = {}

            for key, value in list(logs.items()):
                if key not in H:
                    H[key] = [value]
                else:
                    H[key].append(value)

            pd.DataFrame(H).to_csv(os.path.join(serialization_dir, "history.csv"),
                                   index=False)

        scheduler_config = config.get('learning_rate_scheduler', None)
        if scheduler_config is not None:
            scheduler = ReduceLROnPlateau(monitor='val_acc',
                                          patience=scheduler_config['patience'],
                                          factor=scheduler_config['factor'],
                                          mode=scheduler_config['mode'],
                                          verbose=True)
            callbacks.append(scheduler)

        setattr(time_callback, "t", time.time())
        callbacks.append(LambdaCallback(on_epoch_end=time_callback))
        callbacks.append(LambdaCallback(on_epoch_end=print_callback))
        callbacks.append(EarlyStopping(monitor='val_acc',
                                       patience=config['patience']))
        callbacks.append(LambdaCallback(on_epoch_end=history_callback))

        if config['save_best']:
            best_model_path = os.path.join(serialization_dir, "best_model.h5")
            callbacks.append(ModelCheckpoint(filepath=best_model_path,
                                             monitor='val_acc',
                                             save_best_only=True,
                                             save_weights_only=False))

        if config['save_every_epoch']:
            model_path = os.path.join(serialization_dir, "model_{epoch:02d}.h5")
            callbacks.append(ModelCheckpoint(filepath=model_path,
                                             monitor='val_acc',
                                             save_weights_only=False))

        last_trained_epoch = 0
        if continue_training:
            for epoch in range(1, config['num_epochs']+1):
                if not os.path.exists(os.path.join(serialization_dir,
                                                   f'model_{epoch:02d}.h5')):
                    break
                last_trained_epoch = epoch

        if last_trained_epoch:
            model.load_weights(os.path.join(serialization_dir,
                                            f'model_{epoch:02d}.h5'))
        else:
            model = cls.compile_model(
                model=model,
                config=config
            )

        return cls(
            model=model,
            train_stream=train_stream,
            dev_stream=dev_stream,
            callbacks=callbacks,
            initial_epoch=last_trained_epoch,
            num_epochs=config['num_epochs'],
            serialization_dir=serialization_dir
        )

    def train(self):
        logger.info('Training steps: %d' % len(self.train_stream))
        logger.info('Validation steps: %d' % len(self.dev_stream))

        return self.model.fit_generator(generator=self.train_stream,
                                        steps_per_epoch=len(self.train_stream),
                                        validation_data=self.dev_stream,
                                        validation_steps=len(self.dev_stream),
                                        callbacks=self._callbacks,
                                        initial_epoch=self._initial_epoch,
                                        epochs=self._num_epochs,
                                        use_multiprocessing=True,
                                        verbose=True)
