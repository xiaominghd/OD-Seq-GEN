import torch


def prepare(data):
    traject = torch.zeros([len(data), 24]).type(torch.long)
    res = data.values.tolist()
    for i in range(len(data)):
        traject[i] = torch.tensor(res[i][1:])
    return traject



