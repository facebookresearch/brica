# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
sys.path.insert(0, './pytorch-seq2seq/')

from seq2seq.models import EncoderRNN
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.optim import Optimizer
from t2s import SamplingDecoderRNN as DecoderRNN
from t2s.util import dump_agent
from t2s import T2S, Seq2seq
from t2s.evaluator import PolyEvaluator
from t2s.trainer import MirrorTrainer
import argparse
import pickle
import random
from torch import nn
import numpy as np

import torch
import torchtext
from torchtext.data import Field



device = torch.cuda.current_device() if torch.cuda.is_available() else None


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    ### Define params ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=1234, type=int,
                        help='random seed to initialize your model')
    parser.add_argument('--hidden_size', default=8, type=int,
                        help='Size of the hidden layer of the encoder and decoder (default 8)')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size (default 32)')
    parser.add_argument('--n_layers', default=1, type=int,
                        help='number of hidden layers of the encoder and decoder (default 1)')
    parser.add_argument('--n_epochs', default=10, type=int,
                        help='number of training epochs (default 10)')
    parser.add_argument('--teacher_forcing_ratio', default=1.0, type=float,
                        help='teacher_forcing_ratio: teaching forcing ratio (default 1). Only applicable '
                        'when one of the agents is trained.')
    parser.add_argument('--use_attention', default=1, type=int,
                        help='attention')
    parser.add_argument('--polyglot', default=0, type=int,
                        help='how to optimize teacher part')
    parser.add_argument('--generations', default=0, type=int,
                        help='Agent generations in the iterated learning')
    parser.add_argument('--data_path_prefix', default='./data/Iconic_LessCounting/iconicity_markers',
                        help='where to find the supervised training set for the student and the teacher')
    parser.add_argument('--pretrain_agent', default=0, type=int,
                        help="Toggle this option to pretrain a model as a S2S model")
    parser.add_argument('--tied', default=1, type=int,
                        help="""Integer 0 or 1. If 1, tie the encoder's input embedding and the decoder's
                                outputs matrix""")
    parser.add_argument('--save_model_path', default='model',
                        help='where to save the model')
    parser.add_argument('--init_A1', type=str,
                        help='Load a pre-trained model')
    parser.add_argument('--init_A1_from_A2', type=str,
                        help='Load a pre-trained model')
    parser.add_argument('--max_len', default=30, type=int,
                        help="Maximum length of the sequences")
    parser.add_argument('--explosion_train', default=20, type=int,
                        help="During training, how many instructions to sample")
    parser.add_argument('--explosion_eval', default=120, type=int,
                        help="Maximal number of permutations for instructions")
    parser.add_argument('--eval', default=None, type=str,
                        help="Evaluate the A12 agent (use with --init_A1 or --init_A1_from_A2)")
    parser.add_argument('--no_dev_eval', default=0, type=int,
                        help="Disable evaluation during training")
    parser.add_argument('--no_test_eval', default=1, type=int,
                        help="Disable evaluation during training")

    args = parser.parse_args()
    print(args, flush=True)

    use_attention = args.use_attention == 1
    tied = args.tied == 1
    polyglot = args.polyglot == 1

    if polyglot and not args.pretrain_agent:
        assert False, "Shouldn't use polyglot when not pre-training"

    set_seed(args.random_seed)

    language = args.data_path_prefix.split('/')[-1]
    save_model_path = f'./experiment/models/{args.save_model_path}_attention{args.use_attention}_hidden{args.hidden_size}_batch{args.batch_size}_epoch{args.n_epochs}_tied{args.tied}_seed{args.random_seed}.p'
    teacher_train_path = f'{args.data_path_prefix}_teacher/train/action_instruction.txt'
    teacher_dev_path = f'{args.data_path_prefix}_teacher/dev/action_instruction.txt'
    teacher_test_path = f'{args.data_path_prefix}_teacher/test/action_instruction.txt'

    all_dataset_vocab = './data/Iconic_LessCounting/vocabulary.txt'
    field = Field(preprocessing=lambda x: [
                  '<sos>'] + x + ['<eos>'], unk_token=None, batch_first=True, include_lengths=True, pad_token='<pad>')

    vocab = torchtext.data.TabularDataset(
        path=all_dataset_vocab, format='tsv',
        fields=[('src', field), ('tgt', field)]
    )

    field.build_vocab(vocab, max_size=50000)

    teacher_train = torchtext.data.TabularDataset(
        path=teacher_train_path, format='tsv',
        fields=[('src', field), ('tgt', field)]
    )
    teacher_dev = torchtext.data.TabularDataset(
        path=teacher_dev_path, format='tsv',
        fields=[('src', field), ('tgt', field)]
    )
    teacher_test = torchtext.data.TabularDataset(
        path=teacher_test_path, format='tsv',
        fields=[('src', field), ('tgt', field)]
    )
    print("Vocab: {}".format(field.vocab.stoi), flush=True)

    bidirectional = False
    rnn_cell = 'lstm'

    def get_seq2seq():
        decoder = DecoderRNN(len(field.vocab.stoi), args.max_len,
                             args.hidden_size * 2 if bidirectional else args.hidden_size,
                             n_layers=args.n_layers, rnn_cell=rnn_cell,
                             input_dropout_p=0.0, dropout_p=0.0, use_attention=use_attention,
                             bidirectional=bidirectional,
                             eos_id=field.vocab.stoi['<eos>'], sos_id=field.vocab.stoi['<sos>']).to(device)

        if tied:
            # compatibility with the older code
            nn.init.normal_(decoder.out.weight)

        encoder = EncoderRNN(len(field.vocab.stoi), args.max_len, args.hidden_size,
                             input_dropout_p=0.0, dropout_p=0.0,
                             n_layers=args.n_layers, bidirectional=bidirectional,
                             rnn_cell=rnn_cell, variable_lengths=True,
                             embedding=(decoder.out.weight if tied else None)).to(device)

        return Seq2seq(encoder, decoder)

    if args.init_A1:
        with open(args.init_A1, "rb") as fin:
            m = pickle.load(fin)
        if hasattr(m, "A1"):
            A1 = m.A1
            print('Loaded A1 as submodel')
        else:
            A1 = m
        A1.to(device)
    else:
        A1 = get_seq2seq()

    if args.init_A1_from_A2:
        with open(args.init_A1_from_A2, "rb") as fin:
            A1 = pickle.load(fin).A2.to(device)
        print('Loaded A1 as an A2 submodel')
    A1.flatten_parameters()

    weight = torch.ones(len(field.vocab.stoi), device=device)
    pad = field.vocab.stoi['<pad>']
    loss = NLLLoss(weight, pad)

    train_dataset = teacher_train
    dev_dataset = teacher_dev
    test_dataset = teacher_test

    if args.eval is not None:
        evaluator = PolyEvaluator(
            loss=loss, explosion_rate=args.explosion_eval, batch_size=2048, polyglot=polyglot)
        with open(args.eval, "rb") as fin:
            model = pickle.load(fin)
        eval_results = evaluator.evaluate(model, dev_dataset)
        dev_loss, teacher_accuracy, student_accuracy = eval_results
        log_msg = "Dev %s: %.4f, Accuracy Teacher: %.4f, Accuracy Student: %.4f" % (
            loss.name, dev_loss, teacher_accuracy, student_accuracy)
        print(log_msg, flush=True)

    def train_model(m, poly, pretraining):
        m.train()

        optimizer = Optimizer(torch.optim.Adam(
            m.parameters(), amsgrad=True), max_grad_norm=5)

        t = MirrorTrainer(loss=loss, batch_size=args.batch_size,
                          checkpoint_every=100,
                          expt_dir="./experiments", pretraining=pretraining,
                          polyglot=poly, explosion_train=args.explosion_train, explosion_eval=args.explosion_eval)
        m = t.train(m, train_dataset,
                    n_epochs=args.n_epochs,
                    dev_data=(None if args.no_dev_eval == 1 else dev_dataset),
                    test_data=(None if args.no_test_eval ==
                               1 else test_dataset),
                    optimizer=optimizer,
                    teacher_forcing_ratio=args.teacher_forcing_ratio,
                    resume=False)
        return m


    def dump(agent, path):
        return dump_agent(agent,
                 torchtext.data.BucketIterator(dataset=dev_dataset, batch_size=1024, sort=True, sort_within_batch=True,
                       sort_key=lambda x: len(x.src), device=("cuda" if torch.cuda.is_available() else "cpu"), repeat=False),
                 path,
                 field)

    if args.pretrain_agent:
        A1 = train_model(A1, polyglot, pretraining=True)
        evaluator = PolyEvaluator(
            loss=loss, explosion_rate=args.explosion_eval, batch_size=2048, polyglot=polyglot)

        with open(save_model_path, 'wb') as fout:
            pickle.dump(A1, fout)
            print(f"Saved model to {save_model_path}")
        dump_path = f"{save_model_path}.dump"
        dump(A1, dump_path)
        print(f"Saved model dump to {dump_path}")

    if args.generations > 0:
        for gen in range(1, args.generations + 1):
            set_seed(args.random_seed + gen)
            print("*" * 10, f"Starting generation #{gen}", "*" * 10)
            A2 = get_seq2seq()
            A2.flatten_parameters()
            t2s = T2S(A1, A2)
            t2s = train_model(t2s, poly=False, pretraining=False)
            evaluator = PolyEvaluator(
                loss=loss, explosion_rate=args.explosion_eval, batch_size=2048, polyglot=False)
            eval_results = evaluator.evaluate(t2s, dev_dataset)
            dev_loss, teacher_accuracy, student_accuracy = eval_results
            log_msg = "Dev %s: %.4f, Accuracy Teacher: %.4f, Accuracy Student: %.4f" % (
                loss.name, dev_loss, teacher_accuracy, student_accuracy)
            print(log_msg, flush=True)
            evaluator = PolyEvaluator(
                loss=loss, explosion_rate=args.explosion_eval, batch_size=2048, polyglot=True)
            eval_results = evaluator.evaluate(t2s.A2, dev_dataset)
            dev_loss, teacher_accuracy, student_accuracy = eval_results
            log_msg = "Dev %s: %.4f, Accuracy Teacher GT: %.4f, Accuracy Student GT: %.4f" % (
                loss.name, dev_loss, teacher_accuracy, student_accuracy)
            print(log_msg, flush=True)
            name = f"{save_model_path}.iteration_{gen}"
            with open(name, 'wb') as fout:
                pickle.dump(t2s, fout)
                print(f"Saved model to {name}")
            dump_path = f"{name}.dump"
            dump(t2s.A2, dump_path)
            print(f"Saved model dump to {dump_path}")
            A1 = t2s.A2
