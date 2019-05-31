# BRICA: Bias Research In Communicating Agents

This repo contains the code which we used for the experiments reported in our ACL 2019 paper, titled "Word-order biases in deep-agent emergent communication", [arxiv](https://arxiv.org/abs/1905.12330).

# Installation
 * Cloning the repo:
    `git clone --recursive git@github.com:facebookresearch/brica.git && cd brica`
 * Creating & activating a conda environment:
    `conda create -n brica_env python=3.6 && conda activate brica_env`
 * Installing the requirements:
    `pip install -r requirements.txt`
 * Now tests should pass:
    `pytest`

# How the repo is organized
 * `data/Iconic_LessCounting/` contains the training/validation/test data splits for the artificial languages that are used in the paper.
 Each artificial language reflects or violates various natural language trends, such as the
tendency to avoid redundancy (e.g. `iconicity_teacher` vs `iconicity_markers_teacher`);
 * `pytorch-seq2seq/` is a git submodule containing a 3rd party seq2seq [framework](https://github.com/IBM/pytorch-seq2seq/) based on
 top of [pytorch](https://pytorch.org) which is used in the experiments;
 * `t2s/` and `train.py` contain the actual logic implementation.

# Training

Training of the agents is split in two distinct scenarios:
 * *pre-training*: training of a single agent in separation. This training procedure allows
to study how fast agents learn a particular language, assuming that the faster it is, the easier its
properties are for the agent;
 * *iterated training*: training of the agents in an iterated setup, where an agent is trained in combination
 with another parent agent. After convergence, an agent is fixed and used as a parent to train the next child.
 Using this procedure, we can study the diachronic persistence of different language properties.

An example of a command that would pre-train an agent:
`python train.py --pretrain_agent=1 --n_epochs=10 --no_dev_eval=0 --no_test_eval=0`

This agent can be either loaded for the iterated learning or one can start iterated learning from training the first agent:

`python train.py --pretrain_agent=1 --no_dev_eval=0 --no_test_eval=0 --n_epochs=10 --hidden_size=16 --generations=1`
where `generations` specify the number of iterations in iterated learning.

### Other useful parameters are listed below.
Model specification:
  * `hidden_size`: size of the hidden layer (default is 32),
  * `max_len`: maximal length of an utterance (default is 30),
  * `n_layers`: number of layers in a recurrent unit (default is 1),
  * `use_attention`: whether the decoder uses an attention (default is 1),
  * `tied`: whether the input embedding layer of encoder and the output embedding layer of the decoder are tied (default is 1).
  * `polyglot`: training an agent speaking the free-order language (a _polyglot_),


Training parameters:
  * `num_epochs`: number of training epochs (default is 10),
  * `batch_size`: batch size (default is 32),
  * `data_path_prefix`: the path for the language data (default is `./data/Iconic_LessCounting/iconicity_markers`),
  * `teacher_forcing_ratio`: sets the teacher forcing ratio (default is 1.0),
  * `generations`: the number of iterative learning generations.


Other:
  * `random_seed`: sets the random seed,
  * `explosion_eval`/`explosion_train`: specify the number of samples from a teacher that are used test/training time,
  respectively (defaults are 120 and 20),
  * `init_A1` initializes the teacher from a saved checkpoint,
  * `init_A1_from_A2` initializes the teacher from a student agent from a checkpoint,
  * `save_model_path` set where a trained model would be persisted,
  * `no_test_eval/no_dev_eval` if set to 1, disable validation of the model during training.



# Citation
If you find this code or the ideas in the paper useful in your research, please consider citing the paper [arxiv](https://arxiv.org/abs/1905.12330):
```
@inproceedings{Chaabouni2019,
    title={Word-order biases in deep-agent emergent communication},
    author={Chaabouni, Rahma and Kharitonov, Eugene and Lazaric, Alessandro and Dupoux, Emmanuel and Baroni, Marco},
    booktitle={ACL},
    year={2019}
}
```


# License

BRICA is CC-BY-NC licensed, as found in the LICENSE file.
