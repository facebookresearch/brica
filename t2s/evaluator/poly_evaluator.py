# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function, division

from ..util import repeat_explode, pretrain_explode
from seq2seq.loss import NLLLoss
import seq2seq
import torch
import torchtext

import sys
sys.path.insert(0, 'pytorch_seq2seq')


class PolyEvaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), explosion_rate=120, batch_size=1024, polyglot=False):
        self.loss = loss
        self.batch_size = batch_size
        self.polyglot = polyglot
        self.explosion_rate = explosion_rate

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        loss = self.loss
        loss.reset()
        student_match, teacher_match = 0, 0
        student_total, teacher_total = 0, 0
        device = torch.device("cuda") if torch.cuda.is_available() else None

        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        def eval_one_way(m, src, src_len, dst):
            decoder_outputs, decoder_hidden, other = m(src, src_len, dst)

            for step, step_output in enumerate(decoder_outputs):
                target = dst[:, step + 1]
                loss.eval_batch(step_output.view(dst.size(0), -1), target)

            match, total = 0, 0
            seqlist = other['sequence'][1:]  # cut <sos>
            predictions = torch.stack(seqlist).squeeze(2).permute(1, 0)
            for i in range(src.size(0)):
                total += 1
                target = dst[i, 1:]
                non_padding = target.ne(pad)
                correct = predictions[i].view(-1).eq(
                    target).masked_select(non_padding).sum().item()
                len_non_padding = non_padding.sum().item()
                if correct == len_non_padding:
                    match += 1
            return total, match

        def eval_polyglot_T(m, src, src_len, tgt, tgt_exploded, instance_ids):
            "A bit tricky, as we want to allow any valid output out of possible permutations"
            decoder_outputs, _, other = m(src, src_len, tgt)
            seqlist = other['sequence']
            seqlist = torch.stack(seqlist)
            output_seq = seqlist.squeeze(2).permute(1, 0)

            acc = [0.0 for _ in range(src.size(0))]
            for example_id in range(src.size(0)):
                # get the possible target candidates
                candidate_index = [i for i in range(
                    len(instance_ids)) if instance_ids[i] == example_id]
                for index in candidate_index:
                    if acc[example_id] == 1:
                        # already matched, no point in comparing
                        continue
                    target_candidate = tgt_exploded[index]
                    non_padding = target_candidate.ne(pad)

                    correct = output_seq[example_id, :].view(-1).eq(
                        target_candidate).masked_select(non_padding).sum().item()
                    len_non_padding = non_padding.sum().item()
                    if correct == len_non_padding:
                        acc[example_id] = 1
            return len(acc), sum(acc)

        with torch.no_grad():
            for batch in batch_iterator:
                input_variable, input_lengths = getattr(
                    batch, seq2seq.src_field_name)
                target_variable = getattr(batch, seq2seq.tgt_field_name)

                if hasattr(model, "A2"):
                    A1 = model.A1
                    exploded_input_variable, exploded_input_lengths, src_ids = repeat_explode(
                        input_variable, input_lengths, n_times=self.explosion_rate)

                    teacher_decoder_outputs, _, teacher_other = A1(
                        exploded_input_variable, exploded_input_lengths, None, 0.0)
                    max_len = len(teacher_other['sequence'])

                    A1_target_variable = torch.stack(teacher_other['sequence']).squeeze(2).permute(1, 0)
                    t_out_lengths = model.get_lengths(
                        A1_target_variable, A1.decoder.eos_id).add_(1.0).clamp_(max=max_len)

                    A1_target_variable = torch.stack(
                        teacher_other['sequence']).squeeze(2).permute(1, 0)
                    for i in range(A1_target_variable.size(0)):
                        l = t_out_lengths[i]
                        A1_target_variable[i, l:] = pad

                    A2 = model.A2
                    _total, _match = eval_one_way(
                        A2, A1_target_variable, t_out_lengths, exploded_input_variable)
                    student_total += _total
                    student_match += _match

                    _total, _match = eval_polyglot_T(
                        A2, input_variable, input_lengths, None, A1_target_variable, src_ids)

                    teacher_total += _total
                    teacher_match += _match
                else:
                    A1 = model
                    exploded_input, exploded_input_length, (exploded_target, exploded_target_length), src_ids = \
                        pretrain_explode(input=input_variable, input_length=input_lengths, target_variable=target_variable,
                                         polyglot=self.polyglot, sos=tgt_vocab.stoi[
                                             '<sos>'], eos=tgt_vocab.stoi['<eos>'],
                                         pad=pad, n_samples=self.explosion_rate, sample=True)
                    _total, _match = eval_one_way(
                        A1, exploded_target, exploded_target_length, exploded_input)
                    student_total += _total
                    student_match += _match
                    _total, _match = eval_polyglot_T(
                        A1, input_variable, input_lengths, target_variable[0], exploded_target, src_ids)

                    teacher_total += _total
                    teacher_match += _match

        student_accuracy, teacher_accuracy = student_match / \
            student_total, teacher_match / teacher_total
        return loss.get_loss(), teacher_accuracy, student_accuracy
