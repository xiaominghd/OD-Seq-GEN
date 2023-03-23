import random

import numpy as np
import torch

# data = np.load("../data/train_data.pkl", allow_pickle=True)


def prepare_pretrain(x: list):
    traject = [t for [t, _] in x]
    random.shuffle(traject)
    inp = np.zeros((len(traject), len(traject[0]) - 1))
    target = np.zeros((len(traject), len(traject[0]) - 1))
    for i in range(len(traject)):
        inp[i] = traject[i][:-1]
        target[i] = traject[i][1:]

    return torch.Tensor(inp).type(torch.long), torch.Tensor(target).type(torch.long)


def prepare_discriminator(real: list, fake: list):
    n = len(real)
    real_t = [t for [t, _] in real]
    fake_t = fake
    real_t.extend(fake_t)
    state = np.zeros((len(real_t), len(real_t[0]) - 1))
    action = np.zeros((len(real_t), len(real_t[0]) - 1))
    for i in range(len(real_t)):
        state[i] = real_t[i][:-1]
        action[i] = real_t[i][1:]
    index = np.array([0] * len(real_t))
    index[:n] = [1] * n
    shuffle = np.random.permutation(np.array(range(0, len(real_t), 1)))
    state = state[shuffle]
    action = action[shuffle]
    target = index[shuffle]
    return torch.tensor(state).type(torch.long), torch.tensor(action).type(torch.long), torch.Tensor(target)


