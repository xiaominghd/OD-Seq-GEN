import numpy as np


def gen_reward(model, x, sample_num, dis):

    reward = []

    for i in range(sample_num):

        j = 0
        mid = []

        while x[j + 1][0] != 0:

            data = x[:j + 1].view(1, -1, 3)
            sample = model.sample(data)
            re = dis.reward(sample).data[0].numpy()
            mid.append(re)
            j += 1

        re = dis.reward(x.view(1, -1, 3)).data[0].numpy()
        mid.append(re)
        reward.append(mid)

    return np.sum(np.array(reward), axis=0) / sample_num


