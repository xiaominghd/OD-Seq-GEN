import numpy as np
import pickle
real = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/model/Movsim/data/HaiNan/dispre.data"
with open(real, 'r') as f:
    data = []
    for line in f.readlines():
        line = line.strip("\n").split(" ")
        line = list(map(lambda x:int(x), line))
        data.append(line)

with open('fake_data.pkl', 'wb') as f:
    pickle.dump(data, f)
