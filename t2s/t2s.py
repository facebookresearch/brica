# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn


class T2S(nn.Module):
    def __init__(self, A1, A2):
        super(T2S, self).__init__()
        self.A1 = A1
        self.A2 = A2

    def get_lengths(self, sequence, eos_id):
        eos = sequence.eq(eos_id)
        # eos contains ones on positions where <eos> occur in the outputs, and zeros otherwise
        # eos.cumsum(dim=1) would contain non-zeros on all positions after <eos> occurred
        # eos.cumsum(dim=1) > 0 would contain ones on all positions after <eos> occurred
        # (eos.cumsum(dim=1) > 0).sum(dim=1) equates to the number of timestamps that happened after <eos> occured (including it)
        # eos.size(1) - (eos.cumsum(dim=1) > 0).sum(dim=1) is the number of steps before eos took place
        lengths = eos.size(1) - (eos.cumsum(dim=1) > 0).sum(dim=1)

        return lengths

    def forward(self, input_variable, input_lengths, target_variable, teacher_forcing_ratio=0.0, presorted=False):
        # turn off sampling in the teacher or in the student
        # when needed.
        A1 = self.A1
        with torch.no_grad():
            teacher_decoder_outputs, _, teacher_other = A1(
                input_variable, input_lengths, None, 0.0, presorted=presorted)

        sequence_tensor = torch.stack(teacher_other['sequence']).squeeze(2).permute(1, 0)

        t_out_lengths = self.get_lengths(
            sequence_tensor, A1.decoder.eos_id)

        # NOTE: we increase len by 1 so that the final <eos> is also
        # fed into the student. At the same time, it might be the case that
        #  the teacher never produced <eos>. In tat case, we cap length.
        max_len = len(teacher_other['sequence'])
        t_out_lengths.add_(1.0).clamp_(max=max_len)

        student_decoder_outputs, _, student_other = self.A2(sequence_tensor, t_out_lengths, target_variable=target_variable,
                                                            teacher_forcing_ratio=teacher_forcing_ratio)
        student_other['teacher_decoder'] = teacher_other['sequence']
        student_other['teacher_decoder_outputs'] = teacher_decoder_outputs
        student_other['teacher_dict'] = teacher_other
        student_other['teacher_lengths'] = t_out_lengths

        return student_decoder_outputs, None, student_other
