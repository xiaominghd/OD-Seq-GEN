import random
import torch
import pandas as pd


def choice(data, num=60000):
    """""""""
    在轨迹数据集中选择从早到晚的轨迹
    """""""""
    traject = []
    mid = []
    for i in range(num - 2):

        if data.iloc[i]['id'] == data.iloc[i + 1]['id']:
            mid.append([data.iloc[i]['gps2id'], data.iloc[i]['start_hour'], data.iloc[i]['duration']])

        else:
            mid.append([data.iloc[i]['gps2id'], data.iloc[i]['start_hour'], data.iloc[i]['duration']])
            traject.append(mid)
            mid = []

    random.shuffle(traject)  # 生成随机索引

    return traject


# def choice_single_dis(data, num):
#     traject = torch.zeros(num, 3)
#     for i in range(num):
#         traject[i] = torch.tensor([data.iloc[i]["gps2id"], data.iloc[i]["start_hour"], data.iloc[i]["duration"]])
#     return traject


# 生成轨迹数据
def prepare_pretrain(data):
    data_len = len(data)

    inp = torch.zeros(data_len, 10, 3).type(torch.long)
    target = torch.zeros(data_len, 10, 3).type(torch.long)
    real = torch.zeros(data_len, 10, 3).type(torch.long)

    for i in range(len(data)):

        real[i, :len(data[i]), :] = torch.tensor(data[i])

        if len(data[i]) != 1:
            inp[i, :len(data[i]) - 1, :] = torch.tensor(data[i][:-1])
            target[i, :len(data[i]) - 1, :] = torch.tensor(data[i][1:])

    return inp, target, real


def prepare_dis(pos_sample, neg_sample):
    inp = torch.cat([pos_sample, neg_sample], dim=0)
    target = torch.zeros(inp.size()[0], 1)
    target[:pos_sample.size()[0], 0] = 1
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    return inp.type(torch.long), target


def data2csv(model, inp, file):
    """""""""
    通过model，从inp开始对数据进行采样，并将采样之后的数据输出到file文件当中
    输入:

    inp :          Tensor[[[]]]           num_samples * inp_seq_len * 3

    """""""""
    samples = model.sample(inp).numpy()

    id = []
    grid = []
    start = []
    duration = []

    m = 0

    for s in samples:

        m += 1
        for i in range(len(s)):

            if s[i][0] == 0:

                break
            else:

                id.append(m)
                grid.append(s[i][0])
                start.append(s[i][1])
                duration.append(s[i][2])

    data = {'id': id, 'gps2id': grid, 'start_hour': start, 'duration': duration}

    df = pd.DataFrame(data)  # 转化为dataframe

    df.to_csv(file)
