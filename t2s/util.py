# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import itertools
import random
import json


def repeat_explode(input, input_length, n_times):
    """
    >>> input, input_length  = torch.tensor([[5, 1, 2, 3, 4, 0]]), torch.tensor([5])
    >>> exploded_input, exploded_input_length, src_id = repeat_explode(input=input, input_length=input_length, n_times=3)
    >>> exploded_input
    tensor([[5, 1, 2, 3, 4, 0],
            [5, 1, 2, 3, 4, 0],
            [5, 1, 2, 3, 4, 0]])
    >>> exploded_input_length
    tensor([5, 5, 5])
    >>> src_id
    [0, 0, 0]
    """
    new_input, new_input_length = [], []
    src_id = []

    for i in range(input.size(0)):
        new_input.extend([input[i, :]] * n_times)
        new_input_length.extend([input_length[i]] * n_times)
        src_id.extend([i] * n_times)

    device = input.device
    new_input = torch.stack(new_input).to(device)
    new_input_length = torch.tensor(new_input_length, device=device)

    return new_input, new_input_length, src_id


def pretrain_explode(input, input_length, target_variable, polyglot, sos, eos, pad, sample, n_samples):
    """
    Batch explosion logic; makes it possible to train an agent on samples from another agent.
    Examples:
    >>> input, input_length  = torch.tensor([[5, 1, 2, 3, 4, 0]]), torch.tensor([5])
    >>> target, target_length = torch.tensor([[5, 100, 200, 300, 400, 500, 600, 4, 0, 0]]), torch.tensor([8])
    >>> random.seed(7)
    >>> exploded = pretrain_explode(input=input, input_length=input_length, target_variable=(target, target_length), polyglot=True, sos=5, eos=4, pad=0, n_samples=3, sample=True)
    >>> exploded_input, exploded_input_length, (exploded_target, exploded_target_length), src_ids = exploded
    >>> exploded_input
    tensor([[5, 1, 2, 3, 4, 0],
            [5, 1, 2, 3, 4, 0],
            [5, 1, 2, 3, 4, 0]])
    >>> exploded_input_length
    tensor([5, 5, 5])
    >>> exploded_target
    tensor([[  5, 100, 200, 300, 400, 500, 600,   4,   0,   0],
            [  5, 100, 200, 300, 400, 500, 600,   4,   0,   0],
            [  5, 400, 500, 600, 100, 200, 300,   4,   0,   0]])
    >>> exploded_target_length
    tensor([8, 8, 8])
    >>> src_ids
    [0, 0, 0]
    >>> # now w/o sampling; all possible permutations
    >>> exploded = pretrain_explode(input=input, input_length=input_length, target_variable=(target, target_length), polyglot=True, sos=5, eos=4, pad=0, n_samples=6, sample=False)
    >>> exploded_input, exploded_input_length, (exploded_target, exploded_target_length), src_ids = exploded
    >>> exploded_target
    tensor([[  5, 100, 200, 300, 400, 500, 600,   4,   0,   0],
            [  5, 400, 500, 600, 100, 200, 300,   4,   0,   0]])
    >>> src_ids
    [0, 0]
    """
    new_input, new_length, new_target, new_target_length, src_ids = [], [], [], [], []

    batch_size = input.size(0)
    target, target_length = target_variable
    np_target = target.cpu().numpy()
    max_len = target.size(1)

    for i in range(batch_size):
        l = target_length[i].item()
        grouped = np_target[i, 1:l-1].reshape(-1, 3)
        n_trigram = grouped.shape[0]
        all_permutations = list(itertools.permutations(
            range(n_trigram))) if polyglot else [range(n_trigram)]
        selected_permutations = random.choices(
            all_permutations, k=n_samples) if sample else all_permutations

        for permutation in selected_permutations:
            permutation = grouped[permutation, :].reshape(-1)
            new_input.append(input[i, :])
            new_length.append(input_length[i])
            permutation = [sos] + permutation.tolist() + [eos] + \
                [pad] * (max_len - l)
            new_target.append(permutation)
            new_target_length.append(l)
            src_ids.append(i)

    device = input.device
    new_input = torch.stack(new_input).to(device)
    new_length = torch.tensor(new_length, device=device)
    new_target_length = torch.tensor(new_target_length, device=device)
    new_target = torch.tensor(new_target, device=device)

    return new_input, new_length, (new_target, new_target_length), src_ids


def cut_after_eos(seq):
    try:
        p = seq.index("<eos>")
        out_seq = seq[:p+1]
    except ValueError:
        out_seq = seq
    return out_seq


class LangStats:
    def __init__(self):
        self.language_stats = {}

    def push_stat(self, terms):
        length = self.get_length(terms)
        name = self.get_language_name(terms)

        if length not in self.language_stats:
            self.language_stats[length] = {}
        self.language_stats[length][name] = self.language_stats[length].get(
            name, 0) + 1

    def get_length(self, terms):
        # minus <sos>, <eos>
        return len(terms) - 2

    def get_language_name(self, terms):
        name = []
        for term in terms:
            if term in ['first', 'then', 'finally'] + [f'M{i}' for i in range(6)]:
                name.append(term)
        return '-'.join(name)

    def get_json(self):
        result = {}
        for key, distr in self.language_stats.items():
            per_length = list(distr.items())
            per_length = sorted(per_length, key=lambda x: x[1], reverse=True)
            result[key] = per_length
        return json.dumps(result)


def dump_agent(A, iterator, output_file, field, instruction_explosion_rate=10):
    def id_to_text(ids): return [field.vocab.itos[x.item()] for x in ids]
    stats = LangStats()

    with torch.no_grad(), open(output_file, 'w') as log:
        batch_generator = iterator.__iter__()
        src = []
        for batch in batch_generator:
            tasks = [(batch.src, instruction_explosion_rate, "a->i"),
                     (batch.tgt, 1, "i->a")]
            for ((src, length), explosion_rate, name) in tasks:
                src, length, src_id = repeat_explode(
                    src, length, explosion_rate)
                _1, _2, other = A.forward(src, length, None, 0.0)
                out = torch.stack(other['sequence']).squeeze(2).permute(1, 0)
                prev_src_id = src_id[0]
                for i in range(src.size(0)):
                    if src_id[i] != prev_src_id:
                        prev_src_id = src_id[i]
                        print("*"*20, file=log)

                    src_seq = cut_after_eos(id_to_text(src[i, :]))
                    out_seq = cut_after_eos(id_to_text(out[i, :]))
                    print(src_seq, "->", out_seq, file=log)
                    if name == "a->i":
                        stats.push_stat(out_seq)
            print("*"*20, file=log)
        print("-"*20, file=log)
        print(stats.get_json(), file=log)
