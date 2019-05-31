# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

class EarlyStopping_NoImprovement(object):
    def __init__(self,
                 min_delta=1e-5,
                 patience=5):
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_acc = 0
        self.stopped_epoch = 0
        self._stop_training = False

    def on_epoch_end(self, epoch, current_acc):
        if current_acc is None:
            pass
        else:
            if (current_acc - self.best_acc) > self.min_delta:
                self.best_acc = current_acc
                self.wait = 0  # reset
                self._stop_training = False
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self._stop_training = True

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' %
                  self.stopped_epoch, flush=True)


class EarlyStopping_GoodAccuracy(object):
    def __init__(self, patience=5):
        self.patience = patience
        self.wait = 0
        self.best_accuracy = 0.999
        self.stopped_epoch = 0
        self._stop_training = False

    def on_epoch_end(self, epoch, current_acc):
        if current_acc is None:
            pass
        else:
            if current_acc >= self.best_accuracy:
                self.wait += 1
            else:
                self.wait = 0

            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self._stop_training = True

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' %
                  self.stopped_epoch, flush=True)
