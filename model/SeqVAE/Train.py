import math

import torch
import torch.optim as opt


def pretrain(model, data, seqlen, batch_size, Epoch, lr, crite):
    optim = opt.Adam(model.parameters(), lr=lr)
    batch_num = math.floor(len(data) / batch_size)
    inp = torch.zeros([batch_num, batch_size, seqlen]).type(torch.long)
    for i in range(batch_num):
        inp[i] = data[i * batch_size:i * batch_size + batch_size]
    for epoch in range(Epoch):
        mid_loss = 0
        for batch in range(batch_num):
            optim.zero_grad()
            loss = 0
            for i in range(batch_size):
                dec_out, z_mean, z_log = model(inp[batch][i])
                loss += crite(dec_out, z_mean, z_log, inp[batch][i], 5)
            loss.backward()
            optim.step()
            mid_loss += loss
        print("训练次数为{}".format(epoch + 1))
        print("模型的Loss为{}".format(mid_loss / len(data)))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/SVAE_{"
                                           "}.pth".format(epoch))
