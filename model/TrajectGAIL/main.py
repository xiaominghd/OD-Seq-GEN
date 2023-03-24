from model.GAIL import GAIL, Discriminator
from DataLoader.LoadData import prepare_pretrain, prepare_discriminator
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from Train.pretrain import get_value, train_policy
import argparse


def main(opt):
    data = np.load("data/%s/train_data.pkl" % opt.data, allow_pickle=True)
    inp, target = prepare_pretrain(data)
    batch_size = 512
    batch_num = len(inp) // batch_size
    if opt.data == "HaiNan":
        state_dim = 2500
        action_dim = 2500
        hidden = 32
        num_layers = 3
    if opt.data == "GeoLife":
        state_dim = 38000
        action_dim = 38000
        hidden = 32
        num_layers = 3
    gen = GAIL(state_dim=state_dim, action_dim=action_dim, hidden=hidden, num_layers=num_layers)
    dis = Discriminator(state_dim=state_dim, action_dim=action_dim, hidden=hidden, num_layers=num_layers)

    if not opt.pretrain:
        gen.load_state_dict(torch.load("./Result/%s/pretrain_GAIL.pth" % opt.data))
        dis.load_state_dict(torch.load("./Result/%s/pretrain_dis.pth" % opt.data))
    else:
        opt1 = optim.Adam(gen.parameters(), lr=1e-3)
        for epoch in range(opt.epoch):
            mid_loss = 0
            for num in range(batch_num):
                opt1.zero_grad()
                batch_inp = inp[batch_size * num:batch_size * num + batch_size]
                batch_target = target[batch_size * num:batch_size * num + batch_size]
                loss = gen.pretrain_policy(batch_inp, batch_target)
                mid_loss += loss
                loss.backward()
                opt1.step()
            print("训练次数为{}:".format(epoch))
            print("loss为{}:".format(mid_loss / batch_size))
            torch.save(gen.state_dict(), './Result/%s/pretrain_GAIL.pth' % opt.data)
        fake = np.load("data/%s/fake_data.pkl" % opt.data, allow_pickle=True)
        state, action, flag = prepare_discriminator(data, fake)
        opt2 = optim.Adam(dis.parameters(), lr=1e-3)
        batch_num = len(state) // batch_size
        for epoch in range(opt.epoch):
            mid_loss = 0
            for num in range(batch_num):
                opt2.zero_grad()
                state_inp = state[batch_size * num:batch_size * num + batch_size]
                action_inp = action[batch_size * num:batch_size * num + batch_size]
                flag_target = flag[batch_size * num:batch_size * num + batch_size]
                loss = dis.pretrain(state_inp, action_inp, flag_target)
                mid_loss += loss
                loss.backward()
                opt2.step()
            print("训练次数为{}:".format(epoch))
            print("loss为{}:".format(mid_loss / batch_size))
            torch.save(dis.state_dict(), 'Result/%s/pretrain_dis.pth' % opt.data)
        loss_fn = nn.MSELoss()
        opt3 = optim.Adam(gen.value.parameters(), lr=1e-4)
        for epoch in range(opt.epoch):
            mid_loss = 0
            for num in range(batch_num):
                opt3.zero_grad()
                batch_inp = inp[batch_size * num:batch_size * num + batch_size]
                loss = 0
                hidden = gen.state_emb.init_hidden(batch_size=batch_size)
                for i in range(1, batch_inp.size()[1]):
                    x = batch_inp[:, :i]
                    y = batch_inp[:, i]
                    state, hidden = gen.gen_state(batch_inp[:, i - 1], hidden)
                    s = state.detach()
                    value = get_value(dis, gen, s=x, action=y, seq_len=24, gama=0.98)
                    v = value.detach()
                    value_out = gen.get_value(s, y)
                    loss += loss_fn(value_out, v)
                loss.backward()
                opt3.step()
                mid_loss += loss

            print("训练次数为{}:".format(epoch))
            print(mid_loss / batch_size)
            torch.save(gen.state_dict(), "Result/%s/pretrain_GAIL.pth" % opt.data)
    print("已经加载了模型")
    real = np.load("data/%s/train_data.pkl" % opt.data, allow_pickle=True)
    batch_size = 512
    real_batch = len(real) // batch_size
    real_inp, real_target = prepare_pretrain(real)
    opt2 = optim.SGD(gen.value.parameters(), lr=1e-3)
    opt3 = optim.SGD(gen.policy.parameters(), lr=1e-3)
    opt4 = optim.SGD(dis.parameters(), lr=1e-4)
    for epoch in range(opt.epoch):
        print("第{}次对抗训练开始".format(epoch + 1))
        # 先训练值估计器
        print("***************")
        print("开始训练值估计器")
        for mini_epoch in range(3):
            mid_loss = 0
            loss_fn1 = nn.MSELoss()
            for num in range(real_batch):
                loss = 0
                hidden = gen.state_emb.init_hidden(batch_size=batch_size)
                batch_inp = real_inp[batch_size * num: batch_size * (num + 1)]
                opt2.zero_grad()
                for i in range(1, batch_inp.size()[1]):
                    x = batch_inp[:, :i]
                    y = batch_inp[:, i]
                    state, hidden = gen.gen_state(batch_inp[:, i - 1], hidden)
                    s = state.detach()
                    value = get_value(dis, gen, s=x, action=y, seq_len=24, gama=0.98)
                    v = value.detach()
                    value_out = gen.get_value(s, y)
                    loss += loss_fn1(value_out, v)
                mid_loss += loss
                loss.backward()
                opt2.step()
            print("值估计器在{}次训练后".format(mini_epoch + 1))
            print("loss为:{}".format(mid_loss / batch_size))
            print("_________________")
        print("开始训练策略生成器")
        for mini_epoch in range(3):
            mid_loss = 0
            loss = 0
            for num in range(real_batch):
                print("{}/{}".format(num, real_batch))
                opt3.zero_grad()
                batch_inp = real_inp[num * batch_size: num * batch_size + batch_size, 0].view(-1, 1)
                loss += train_policy(gen, batch_inp)
            loss.backward()
            opt3.step()
            mid_loss += loss
            print("策略生成器在{}次训练后".format(mini_epoch + 1))
            print("loss为:{}".format(mid_loss / batch_size))
            print("_________________")
        print("***************")
        print("开始训练判别器")
        fake = gen.sample(real_inp[:, 0].view(-1, 1), 24)
        dis_state, dis_action, dis_target = prepare_discriminator(real, fake, raw=False)
        dis_num = len(dis_target) // batch_size
        for mini_epoch in range(3):
            mid_loss = 0
            for num in range(dis_num):
                opt4.zero_grad()
                batch_state = dis_state[batch_size * num:batch_size * num + batch_size]
                batch_action = dis_action[batch_size * num:batch_size * num + batch_size]
                batch_target = dis_target[batch_size * num:batch_size * num + batch_size]
                loss = dis.pretrain(batch_state, batch_action, batch_target)
                loss.backward()
                opt4.step()
                mid_loss += loss
            print("训练判别器epoch为:{}".format(mini_epoch))
            print("loss为:{}".format(mid_loss / batch_size))
            print("*******************")
        torch.save(gen.state_dict(), 'Result/%s/GAIL_GAN.pth' % opt.data)
        torch.save(dis.state_dict(), 'Result/%s/dis_GAN.pth' % opt.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default=False, action='store_true')
    parser.add_argument('--data', default="HaiNan")
    parser.add_argument('--epoch', default=20)
    opt = parser.parse_args()
    main(opt)
