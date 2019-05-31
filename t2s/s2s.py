# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def get_batch_permutation(lengths, device):
        """
        Returns a permutation and its reverse that turns `lengths` in a sorted
        list in descending order.
        >>> lengths = [4, 1, 0, 100]
        >>> permutation, inverse = Seq2seq.get_batch_permutation(lengths, torch.device("cpu"))
        >>> permutation
        tensor([3, 0, 1, 2])
        >>> rearranged = torch.index_select(torch.tensor(lengths), 0, permutation)
        >>> rearranged
        tensor([100,   4,   1,   0])
        >>> torch.index_select(rearranged, 0, inverse)
        tensor([  4,   1,   0, 100])
        """
        lengths = torch.tensor(lengths, device=device)
        _, rearrange = torch.sort(lengths, descending=True)
        inverse = torch.zeros_like(lengths)
        for i, v in enumerate(rearrange):
            inverse[v] = i
        return rearrange.to(device), inverse.to(device)

    def rearrange_output(self, rearrange, decoder_outputs, other):
        new_other = {}
        new_other[self.decoder.KEY_LENGTH] = [
            other[self.decoder.KEY_LENGTH][i] for i in rearrange]
        new_other[self.decoder.KEY_SEQUENCE] = [torch.index_select(
            s, 0, rearrange) for s in other[self.decoder.KEY_SEQUENCE]]
        new_decoder_outputs = [torch.index_select(
            t, 0, rearrange) for t in decoder_outputs]

        return new_decoder_outputs, new_other

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, presorted=False):
        if not input_lengths is None and not presorted:
            rearrange, inverse = self.get_batch_permutation(
                input_lengths, input_variable.device)
            input_variable = torch.index_select(input_variable, 0, rearrange)
            input_lengths = [input_lengths[i] for i in rearrange]
            if not target_variable is None:
                target_variable = torch.index_select(
                    target_variable, 0, rearrange)

        encoder_outputs, encoder_hidden = self.encoder(
            input_variable, input_lengths)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=F.log_softmax,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        if not input_lengths is None and not presorted:
            decoder, other = self.rearrange_output(
                inverse, result[0], result[2])
            result = decoder, None, other

        batch_size = input_variable.size(0)
        sos_tensor = torch.tensor(
            [self.decoder.sos_id] * batch_size, device=input_variable.device).view(-1, 1)
        # add sos at the start
        key = self.decoder.KEY_SEQUENCE
        result[2][key] = [sos_tensor] + result[2][key]
        return result
