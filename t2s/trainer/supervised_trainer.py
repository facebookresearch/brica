#
# Copyright 2017- IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Below is a modification of the SupervisedTrainer from pytorch-seq2seq/seq2seq/trainer/supervised_trainer.py
# which is adopted from https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/trainer/supervised_trainer.py
# The changes we have introduced relate to: early stopping and progress report logic.

import sys
sys.path.insert(0, './pytorch_seq2seq')

from seq2seq.util.checkpoint import Checkpoint
from seq2seq.optim import Optimizer
from seq2seq.loss import NLLLoss
import seq2seq
import os
import random

import torch
import torchtext
from torch import optim
from .early_stopping import EarlyStopping_NoImprovement




class SupervisedTrainer(object):
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, patience=5):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss

        # set by a subclass
        self.evaluator = None
        self.optimizer = None
        self.checkpoint_every = checkpoint_every

        self.early_stopping_teacher = EarlyStopping_NoImprovement(
            patience=patience)
        self.early_stopping_student = EarlyStopping_NoImprovement(
            patience=patience)

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        raise NotImplementedError('Not implemented!')

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, test_data=None, teacher_forcing_ratio=0):

        epoch_loss_total = 0  # Reset every epoch

        device = torch.device("cuda") if torch.cuda.is_available() else -1

        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs
        print('total steps is equal to ', total_steps)

        step = start_step
        step_elapsed = 0
        epoch = start_epoch
        max_epoch_iteration = n_epochs + 1
        while epoch < max_epoch_iteration:

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths = getattr(
                    batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)

                loss = self._train_batch(input_variables, input_lengths.tolist(
                ), target_variables, model, teacher_forcing_ratio)
                # Record average loss
                epoch_loss_total += loss

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / \
                min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (
                epoch, self.loss.name, epoch_loss_avg)

            time_logging = 1

            if dev_data is not None and (epoch % time_logging) == 0:
                eval_results = self.evaluator.evaluate(model, dev_data)
                assert len(eval_results) == 3

                # if two-hat agent, we have accuracies for the both hats (Student and Teacher)
                dev_loss, teacher_accuracy, student_accuracy = eval_results
                log_msg += ", Dev %s: %.4f, Accuracy Teacher: %.4f, Accuracy Student: %.4f" % (
                    self.loss.name, dev_loss, teacher_accuracy, student_accuracy)

                self.early_stopping_student.on_epoch_end(
                    epoch, student_accuracy)
                self.early_stopping_teacher.on_epoch_end(
                    epoch, teacher_accuracy)

                if self.early_stopping_student._stop_training and self.early_stopping_teacher._stop_training:
                    max_epoch_iteration = self.early_stopping_teacher.stopped_epoch

                model.train(mode=True)

            if test_data is not None and (epoch % time_logging) == 0:
                assert len(eval_results) == 3
                # if two-hat agent, we have accuracies for the both hats (Student and Teacher)
                test_loss, teacher_accuracy, student_accuracy = eval_results
                log_msg += ", Test %s: %.4f, Accuracy Teacher: %.4f, Accuracy Student: %.4f" % (
                    self.loss.name, test_loss, teacher_accuracy, student_accuracy)
            epoch += 1

            print(log_msg, flush=True)

    def train(self, model, data, n_epochs=5,
              resume=False, dev_data=None, test_data=None,
              optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            n_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            test_data (seq2seq.dataset.dataset.Dataset, optional): test Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(
                self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(
                model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(
                    model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self._train_epoches(data, model, n_epochs,
                            start_epoch, step, dev_data=dev_data, test_data=test_data,
                            teacher_forcing_ratio=teacher_forcing_ratio)

        if self.early_stopping_student.stopped_epoch > 0 and self.early_stopping_teacher._stop_training > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' %
                  (max(self.early_stopping_student.stopped_epoch, self.early_stopping_teacher.stopped_epoch)), flush=True)
        return model
