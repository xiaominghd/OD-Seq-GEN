import random
import numpy as np
import pandas as pd
import torch
from generator import Generator

VOCAB_SIZE = 2500
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 24
model = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, use_cuda=None)
model.load_state_dict(torch.load("/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/SanYa/SeqGAN"
                                 "/gen_gan_10.pth"))

grid_list = []
with open(r'/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan/Sanya/real_SanYa_seq.txt') as f:
    for line in f.readlines():
        s = line.strip().split(" ")
        grid_list.append(int(line.split(" ")[0]))


def generate_samples(model, batch_size, generated_num, output_file):  # 输出的samples是一个序列，序列的长度是
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()  # sample出来的是一个长为batch_size的序列
        samples.extend(sample)

    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


generate_samples(model, batch_size=128, generated_num=30000, output_file="/mnt/data/gonghaofeng/deeplearning_project"
                                                                         "/ODseq_GAN_remote/Gen_data/SanYa/SeqGAN"
                                                                         "/gen_seq.txt")
df = pd.DataFrame(columns=['id', 'gps2id', 'start_hour', 'duration'])
data_file = '/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/SanYa/SeqGAN/gen_seq.txt'
with open(data_file, 'r') as f:
    lines = f.readlines()
i = 0
for line in lines:
    l = line.strip().split(' ')
    l = [int(s) for s in l]
    i += 1
    left = 0
    right = 0
    while right < len(l) - 1:
        if l[right] == 0:
            break
        if l[right] != l[right + 1]:
            if right - left > 1:
                df = pd.concat(
                    [df, pd.DataFrame([[i, l[right], left, right - left]],
                                      columns=['id', 'gps2id', 'start_hour', 'duration'])])
            left = right + 1
        right += 1
df.to_csv('/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/SanYa/SeqGAN/seqgan.csv')
