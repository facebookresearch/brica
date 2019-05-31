# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from __future__ import division
import torch

from seq2seq.loss import NLLLoss
from ..evaluator import PolyEvaluator
from ..util import repeat_explode, pretrain_explode

from .supervised_trainer import SupervisedTrainer


class MirrorTrainer(SupervisedTrainer):
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, pretraining=False, polyglot=False, explosion_train=10, explosion_eval=120):
        super(MirrorTrainer, self).__init__(
            expt_dir=expt_dir, loss=loss, batch_size=batch_size, random_seed=random_seed,
            checkpoint_every=checkpoint_every)
        self._trainer = "Mirror Trainer"
        self.pretraining = pretraining
        self.polyglot = polyglot
        self.evaluator = PolyEvaluator(
            explosion_rate=explosion_eval, loss=self.loss, batch_size=512, polyglot=self.polyglot)
        self.explosion_train = explosion_train

    def _one_direction_pass(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio, presorted):
        batch_size = target_variable.size(0)
        outputs, _, other = model(input_variable, input_lengths, target_variable,
                                  teacher_forcing_ratio=teacher_forcing_ratio, presorted=presorted)
        for step, step_output in enumerate(outputs):
            self.loss.eval_batch(step_output.contiguous().view(
                batch_size, -1), target_variable[:, step + 1])
        return outputs, other

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        self.loss.reset()

        sos, eos = None, None
        if hasattr(model, 'A1'):
            sos, eos = model.A1.decoder.sos_id, model.A1.decoder.eos_id
        else:
            sos, eos = model.decoder.sos_id, model.decoder.eos_id

        if self.pretraining:
            input_variable, input_lengths, target_variable, _ = pretrain_explode(input_variable, input_lengths, target_variable,
                                                                                 polyglot=self.polyglot,
                                                                                 sos=sos, eos=eos, pad=0,
                                                                                 n_samples=self.explosion_train, sample=True)

            self._one_direction_pass(
                input_variable, input_lengths, target_variable[0], model, teacher_forcing_ratio, True)
            self._one_direction_pass(
                target_variable[0], target_variable[1], input_variable, model, teacher_forcing_ratio, False)
        else:
            input_variable, input_lengths, _ = repeat_explode(
                input_variable, input_lengths, self.explosion_train)
            _, outputs = self._one_direction_pass(
                input_variable, input_lengths, input_variable, model, teacher_forcing_ratio, True)
            batch_size = input_variable.size(0)
            i_A1 = torch.stack(outputs['teacher_decoder']).squeeze(2).permute(1, 0)
            i_A1_length = outputs['teacher_lengths']
            pad_id = 0

            for i in range(batch_size):
                l = i_A1_length[i]
                i_A1[i, l:] = pad_id
            self._one_direction_pass(
                input_variable, input_lengths, i_A1, model.A2, teacher_forcing_ratio, False)

        model.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        return self.loss.get_loss()
