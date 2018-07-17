# -*- coding: utf-8 -*-


import os
import time
import pandas as pd
import numpy as np
# import cPickle as pickle
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, Callback, ReduceLROnPlateau
from keras.models import load_model
import keras
import keras.backend as K

# # Might misbehave with tensorflow-gpu, make sure u use tensorflow-cpu if using Theano for keras
# try:
#     import tensorflow
# except:
#     pass

import logging
logger = logging.getLogger(__name__)


class LambdaCallbackPickable(LambdaCallback):
    def set_callback_state(self, callback_state={}):
        self.callback_state = callback_state

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['on_epoch_end']
        del state['on_epoch_begin']
        del state['on_batch_end']
        del state['on_train_end']
        del state['on_train_begin']
        del state['on_batch_begin']
        return state

    def __setstate__(self, newstate):
        newstate['on_epoch_end'] = self.on_epoch_end
        newstate['on_train_end'] = self.on_train_end
        newstate['on_epoch_begin'] = self.on_epoch_begin
        newstate['on_train_begin'] = self.on_train_begin
        newstate['on_batch_end'] = self.on_batch_end
        newstate['on_batch_begin'] = self.on_batch_begin
        self.__dict__.update(newstate)


# class DumpTensorflowSummaries(Callback):
#     def __init__(self, save_path):
#         self._save_path = save_path
#         super(DumpTensorflowSummaries, self).__init__()
#
#     @property
#     def file_writer(self):
#         if not hasattr(self, '_file_writer'):
#             self._file_writer = tensorflow.summary.FileWriter(
#                 self._save_path, flush_secs=10.)
#         return self._file_writer
#
#     def on_epoch_end(self, epoch, logs=None):
#         summary = tensorflow.Summary()
#         for key, value in logs.items():
#             try:
#                 float_value = float(value)
#                 value = summary.value.add()
#                 value.tag = key
#                 value.simple_value = float_value
#             except:
#                 pass
#         self.file_writer.add_summary(
#             summary, epoch)

class WarmUp:

    def __init__(self, K, start_lr, peak_lr, end_lr, dataset_size, batch_size, n_epochs):
        self.steps_till_peak = K * (dataset_size // batch_size)
        self.steps_till_end = (n_epochs - K) * (dataset_size // batch_size)
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr

    def __call__(self, batch_number):
        if batch_number <= self.steps_till_peak:
            return self.start_lr + (self.peak_lr - self.start_lr) * (batch_number/self.steps_till_peak)
        else:
            return self.peak_lr + (self.end_lr - self.peak_lr) * ((batch_number-self.steps_till_peak)/self.steps_till_end)


def create_lr_schedule(config, model, dataset_size, batch_size, save_path):
    learning_rate_schedule_type = config.get("lr_schedule_type", "list_of_lists")
    learning_rate_schedule = eval(config['lr_schedule'])
    batch_size = config["batch_sizes"]["train"]
    n_epochs = config["n_epochs"]

    if learning_rate_schedule_type == "reduce_on_plateau":
        return ReduceLROnPlateau(patience=5, verbose=1)
    elif learning_rate_schedule_type == "list_of_lists":
        def lr_schedule(epoch, logs):
            for e, v in learning_rate_schedule:
                if epoch < e:
                    break
            setattr(model.optimizer, "base_lr", v)
            K.set_value(model.optimizer.lr, v)
            logger.info("Set learning rate to {}".format(v))

        return LambdaCallback(on_epoch_begin=lr_schedule, on_epoch_end=lr_schedule)
    elif learning_rate_schedule_type == "warmup":
        cls = LambdaCallbackPickable()
        cls.set_callback_state({"batch_id": np.array([0])})
        warmup = WarmUp(learning_rate_schedule[0], learning_rate_schedule[1], learning_rate_schedule[2],
                        learning_rate_schedule[3],
                        dataset_size,
                        batch_size, n_epochs)

        def lr_schedule(batch, logs):
            batch_id = cls.callback_state['batch_id']
            v = warmup(batch_id[0])
            setattr(model.optimizer, "base_lr", v)
            K.set_value(model.optimizer.lr, v)
            if batch_id % 100 == 0:
                logger.info("Set learning rate to {}".format(v))
            batch_id[0] += 1

        cls.on_batch_begin = lr_schedule
        return cls
    else:
        raise NotImplementedError("Not implemented learning rate schedule")


# train, test, dev are generators
def baseline_training_loop(model, data_and_streams,
                           early_stopping, n_epochs,
                           save_path, config):

    train_num_examples = data_and_streams["data"].num_examples("train")
    train_batch_size = config["batch_sizes"]["train"]

    dev_num_examples = data_and_streams["data"].num_examples("dev")
    dev_batch_size = config["batch_sizes"]["dev"]

    if os.path.exists(os.path.join(save_path, "loop_state.pkl")):
        logger.info("Reloading loop state")
        # model = load_model(os.path.join(save_path, 'loop_state.h5'))
        loop_state = pickle.load(open(os.path.join(save_path, "loop_state.pkl")))
    else:
        loop_state = {'last_epoch_done_id': -1}

    if os.path.exists(os.path.join(save_path, "model.h5")):
        model.load_weights(os.path.join(save_path, "model.h5"))

    callbacks = [ModelCheckpoint(filepath=os.path.join(save_path, "best_model.h5"),
                                 save_best_only=True,
                                 save_weights_only=True)]

    def time_callback(epoch, logs):
        t = getattr(time_callback, "t")
        logs['time_epoch'] = time.time() - t
        setattr(time_callback, "t", time.time())
    setattr(time_callback, "t", time.time())
    callbacks.append(LambdaCallback(on_epoch_end=time_callback))

    #lr schedule
    if config["lr_schedule_type"] != "none":
        callbacks.append(create_lr_schedule(
            config, model, train_num_examples, train_batch_size, save_path))

    def eval_on_test(epoch, logs):
        test_num_examples = data_and_streams["data"].num_examples("test")
        test_batch_size = config["batch_sizes"]["test"]
        logs['test_loss'], logs['test_acc'] = model.evaluate_generator(
                generator=data_and_streams["test"],
                steps=test_num_examples//test_batch_size)
        # print(("Test loss, test accuracy: {}, {}".format(B[0], B[1])))
    callbacks.append(LambdaCallback(on_epoch_end=eval_on_test))

    def print_logs(epoch, logs):
        print()
        for k, v in list(logs.items()):
            logger.info("{}={}".format(k, v))
    callbacks.append(LambdaCallback(on_epoch_end=print_logs))

    if early_stopping == True:
        callbacks.append(EarlyStopping(monitor='val_acc', patience=8))

    def save_history(epoch, logs):
        history_path = os.path.join(save_path, "history.csv")
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

        pd.DataFrame(H).to_csv(os.path.join(save_path, "history.csv"), index=False)
    callbacks.append(LambdaCallback(on_epoch_end=save_history))

    # Uncomment if you have tensorflow installed correctly
    # callbacks.append(DumpTensorflowSummaries(save_path=save_path))
    callbacks.append(ModelCheckpoint(monitor='val_acc',
                                     save_weights_only=False,
                                     filepath=os.path.join(save_path, "model.h5")))

    def save_loop_state(epoch, logs):
        loop_state = {"last_epoch_done_id": epoch}
        # model.save(os.path.join(save_path, 'loop_state.h5'))
        pickle.dump(loop_state, open(os.path.join(save_path, "loop_state.pkl"), "wb"))
    callbacks.append(LambdaCallback(on_epoch_end=save_loop_state))

    logger.info('Training...')
    _ = model.fit_generator(data_and_streams["train"],
                            initial_epoch=loop_state['last_epoch_done_id'] + 1,
                            steps_per_epoch=train_num_examples * config["train_on_fraction"] // train_batch_size,
                            epochs=n_epochs, verbose=1,
                            validation_data=data_and_streams["dev"],
                            use_multiprocessing=True,
                            validation_steps=dev_num_examples // dev_batch_size,
                            callbacks=callbacks)
