import sys

import torch
import Discriminator
import DataLoader
from NewModel import MyModel
import argparse
import pandas as pd
import Train as Train
import torch.optim as optim
from roll_out import gen_reward

parser = argparse.ArgumentParser()
parser.add_argument("--place", type=str, default='SanYa', help="where to generate trajectories")
parser.add_argument("--pretrain_epochs", type=int, default=21, help="number of epochs of pretraining generator")
parser.add_argument("--pretrain_lr", type=int, default=1e-3, help="learning rate of pretraining generator")
parser.add_argument("--pretrain_batch_size", type=int, default=128, help="batch size of pretraining generator")
parser.add_argument("--loc_emb_dim", type=int, default=16, help="number of location embedding")
parser.add_argument("--tim_emb_dim", type=int, default=16, help="number of time embedding")
parser.add_argument("--pos_emb_dim", type=int, default=32, help="number of position embedding")
parser.add_argument("--point_size", type=int, default=2500, help="number of grids")
parser.add_argument("--latent_dim", type=int, default=5, help="dim of latent")
parser.add_argument("--hidden_dim", type=int, default=32, help="dim of GRU hidden")
parser.add_argument("--dis_hidden_dim", type=int, default=16, help="dim of discriminator GRU hidden")
parser.add_argument("--sample_num", type=int, default=8000, help="num of samples")
parser.add_argument("--GAN_epochs", type=int, default=30, help="num of epochs of training GAN")
parser.add_argument("--GAN_dis_lr", type=int, default=1e-4, help="learning rate of GAN training discriminator")

opt = parser.parse_args()
if opt.place == "HaiKou":
    Data_File = '/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan/HaiKou/'
    Gen_Data_File = '/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/HaiKou/OD_seqGAN/'
    Model_File = '/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/HaiKou/OD_seqGAN/'
if opt.place == "SanYa":
    Data_File = '/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan/Sanya/'
    Gen_Data_File = '/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/SanYa/OD_seqGAN/'
    Model_File = '/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/SanYa/OD_seqGAN/'
#
real_data = DataLoader.choice(pd.read_csv(Data_File + "SanYa.csv"), num=40000)
#
# # 预训练生成器
gen = MyModel(loc_emb_dim=opt.loc_emb_dim, tim_emb_dim=opt.tim_emb_dim, pos_emb_dim=opt.pos_emb_dim,
              input_dim=opt.latent_dim, point_size=opt.point_size, hidden_dim=opt.hidden_dim)
gen.load_state_dict(torch.load(Model_File+"gen_S_20.pth"))
# gen_inp, gen_target, _ = DataLoader.prepare_pretrain(real_data)
# Train.pretrain_NLL(model=gen, inp=gen_inp, target=gen_target, Epoch=opt.pretrain_epochs,
#                    lr=opt.pretrain_lr, batch_size=opt.pretrain_batch_size)

# 预训练序列判别器
r_dis = Discriminator.seq_dis(point_size=opt.point_size, loc_emb_dim=opt.loc_emb_dim,
                              tim_emb_dim=opt.tim_emb_dim, input_dim=opt.latent_dim, hidden_dim=opt.dis_hidden_dim)
r_dis.load_state_dict(torch.load(Model_File+"dis_S_20.pth"))
# r_pos = DataLoader.choice(pd.read_csv("/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan"
#                                       "/Sanya/real_SanYa.csv"), num=10000)
# _, _, r_pos_samples = DataLoader.prepare_pretrain(r_pos)
# r_neg = DataLoader.choice(pd.read_csv(r"/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan"
#                                       r"/Sanya/fake_SanYa.csv"), num=10000)
# _,_, r_neg_samples = DataLoader.prepare_pretrain(r_neg)
# #
# r_dis_inp, r_dis_target = DataLoader.prepare_dis(r_pos_samples, r_neg_samples)
#
# Train.train_dis(model=r_dis, inp=r_dis_inp, target=r_dis_target, lr=opt.pretrain_lr,
#                 Epoch=opt.pretrain_epochs, batch_size=opt.pretrain_batch_size, model_name='S')
#
# # 预训练类型准测
# # def trans(data):
# #     x = []
# #     for i in range(len(data)):
# #         for j in range(len(data[i]) - 1):
# #             x.append([data[i][j], data[i][j + 1]])
# #     return torch.tensor(x)
# #
# #
# # loc_emb = Emb.Emb_loc(loc_emb_size=opt.loc_emb_dim, point_size=opt.point_size)
# # n_dis = Discriminator.neighbor_dis(poi_emb=loc_emb, poi_emb_dim=opt.loc_emb_dim,
# #                                    tim_emb_dim=opt.tim_emb_dim, latent_dim=opt.latent_dim)
# #
# # n_pos = trans(DataLoader.choice(pd.read_csv("data/real_HaiKou.csv"), num=20000))
# # n_neg = trans(DataLoader.choice(pd.read_csv("data/fake_HaiKou.csv"), num=20000))
# # n_dis_inp, n_dis_target = DataLoader.prepare_dis(n_pos, n_neg)
# #
# # Train.train_dis(model=n_dis, inp=n_dis_inp, target=n_dis_target, lr=opt.pretrain_lr,
# #                 Epoch=opt.pretrain_epochs, batch_size=opt.pretrain_batch_size, model_name='n_dis')
#
# 对抗训练生成器

GAN_inp, GAN_target, GAN_real = DataLoader.prepare_pretrain(real_data)

gen_opt = optim.Adam(gen.parameters(), lr=1e-3)

DataLoader.data2csv(gen, GAN_inp[:opt.sample_num, 0, :].view(opt.sample_num, -1, 3), Gen_Data_File + "gen_gan_0.csv")

for e in range(5, opt.GAN_epochs):

    print("第{}次训练开始:".format(e + 1))
    print("************")

    for epoch in range(3):

        mid_loss = 0

        for num in range(10):

            gen_opt.zero_grad()
            loss1 = 0

            data1 = GAN_real[num * opt.pretrain_batch_size:(num + 1) * opt.pretrain_batch_size]
            inp1 = GAN_inp[num * opt.pretrain_batch_size:(num + 1) * opt.pretrain_batch_size]
            target1 = GAN_target[num * opt.pretrain_batch_size:(num + 1) * opt.pretrain_batch_size]

            for i in range(len(data1)):

                reward = gen_reward(gen, data1[i], sample_num=8, dis=r_dis)
                hidden = gen.init_hidden()
                j = 0

                while target1[i][j][0] != 0:
                    loc, arr, dur, hidden = gen(inp1[i][j], hidden, torch.tensor(j))
                    loss1 -= (loc[target1[i][j][0]] + arr[target1[i][j][1]] + dur[target1[i][j][2]]) * reward[j]
                    j += 1

            loss1.backward()
            gen_opt.step()
            mid_loss += loss1

        print("生成器第{}次训练后".format(epoch + 1))
        print("loss为{}".format(mid_loss / opt.pretrain_batch_size))
        print("_________________")

        _, _, GAN_pos_sample = DataLoader.prepare_pretrain(
            DataLoader.choice(pd.read_csv(Data_File + "real_SanYa.csv"), num=10000))
        _, _, GAN_neg_sample = DataLoader.prepare_pretrain(DataLoader.choice(
            pd.read_csv(Gen_Data_File + "gen_gan_{}.csv".format(e)), num=10000))

        GAN_dis_inp, GAN_dis_target = DataLoader.prepare_dis(GAN_pos_sample, GAN_neg_sample)

        Train.train_dis(model=r_dis, inp=GAN_dis_inp, target=GAN_dis_target, lr=opt.GAN_dis_lr, Epoch=2,
                        batch_size=opt.pretrain_batch_size, model_name='gan')

        GAN_inp, GAN_target, GAN_real = DataLoader.prepare_pretrain(real_data)
        DataLoader.data2csv(gen, GAN_inp[:opt.sample_num, 0, :].view(opt.sample_num, -1, 3),
                            Gen_Data_File + "gen_gan_{}.csv".format(e + 1))

        if (e + 1) % 5 == 0:
            torch.save(gen.state_dict(), Model_File + "gen_GAN_{}.pth".format(e + 1))
