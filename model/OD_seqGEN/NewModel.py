import torch.nn as nn
import torch
import torch.nn.functional as F


# ResNet网络结构，有全连接的都改成用ResNet来
# 输入：torch.Tensor : 1 * DIM
# 输出：torch.Tensor : 1 * DIM

class ResNet(nn.Module):

    def __init__(self, DIM):
        super(ResNet, self).__init__()
        self.DIM = DIM
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.Dropout(p=0.8)
        )

    def forward(self, x):
        out = self.net(x)
        return x + 0.3 * out


def gen_single(loc, arr, dur, before_tim, before_loc):
    loc = torch.exp(loc)
    loc[before_loc] = 0
    loc_value, loc_index = torch.topk(loc, 5)
    p_loc = loc_index[torch.multinomial(loc_value, num_samples=1)]

    arr = torch.exp(arr)
    arr[:before_tim] = torch.zeros(before_tim)
    p_arr = torch.multinomial(arr, num_samples=1)

    dur = torch.exp(dur)
    dur[24 - p_arr:] = torch.zeros(p_arr)
    p_dur = torch.multinomial(dur, num_samples=1)

    if p_arr + p_dur < 24 and p_arr < 21:
        return torch.tensor([p_loc, p_arr, p_dur])
    else:
        return torch.tensor([0, 0, 0])


class MyModel(nn.Module):

    def __init__(self, loc_emb_dim=16, tim_emb_dim=16, pos_emb_dim=32, input_dim=5, point_size=10000,
                 hidden_dim=32):
        super(MyModel, self).__init__()
        self.loc_emb = nn.Embedding(point_size, loc_emb_dim)
        self.arr_emb = nn.Embedding(24, tim_emb_dim)
        self.dur_emb = nn.Embedding(24, tim_emb_dim)
        self.pos_emb = nn.Embedding(10, pos_emb_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.fc1 = nn.Sequential(
            ResNet(loc_emb_dim),
            ResNet(loc_emb_dim),
            nn.Linear(loc_emb_dim, input_dim)
        )
        self.fc2 = nn.Sequential(
            ResNet(tim_emb_dim),
            ResNet(tim_emb_dim),
            nn.Linear(tim_emb_dim, input_dim)
        )
        self.fc3 = nn.Sequential(
            ResNet(tim_emb_dim),
            ResNet(tim_emb_dim),
            nn.Linear(tim_emb_dim, input_dim)
        )
        self.LM = nn.LayerNorm(self.input_dim * 3)
        self.ODE = nn.Sequential(nn.Linear(input_dim, hidden_dim), ResNet(hidden_dim))
        self.fc4 = nn.Linear(self.hidden_dim, point_size)
        self.fc5 = nn.Linear(self.hidden_dim, 24)
        self.fc6 = nn.Linear(self.hidden_dim, 24)
        self.GRU = nn.GRU(input_size=self.input_dim * 3, hidden_size=self.hidden_dim)

    def init_hidden(self):
        return torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim))

    def forward(self, x, hidden, pos):

        pos = self.pos_emb(pos)

        loc = self.fc1(self.loc_emb(x[0]))
        arr = self.fc2(self.arr_emb(x[1]))
        dur = self.fc3(self.dur_emb(x[2]))

        point = torch.concat([loc, arr, dur], dim=0).view(1, 1, -1)
        point = self.LM(point)

        hidden = hidden + self.ODE(dur).view(1, 1, -1)

        out, hidden = self.GRU(point, hidden)
        next_loc = F.log_softmax(self.fc4(out), dim=2).view(-1)
        next_arr = F.log_softmax(self.fc5(out + pos), dim=2).view(-1)
        next_dur = F.log_softmax(self.fc6(out), dim=2).view(-1)
        return next_loc, next_arr, next_dur, hidden

    def train_NLL(self, inp, target):

        batch_size = inp.size()[0]
        loss_fn = nn.NLLLoss()
        loss = 0

        for i in range(batch_size):

            j = 0
            hidden = self.init_hidden()

            while j < len(target[i]) and target[i][j][0] != 0:
                next_loc, next_arr, next_dur, hidden = self.forward(inp[i][j], hidden, torch.tensor(j))
                loss1 = loss_fn(next_loc, target[i][j][0])
                loss2 = loss_fn(next_arr, target[i][j][1])
                loss3 = loss_fn(next_dur, target[i][j][2])
                loss = loss + loss1 + loss2 + loss3
                j += 1
        return loss / batch_size

    def sample(self, start):

        num_samples = start.size()[0]
        input_seq = start.size()[1]

        samples = torch.zeros([num_samples, 10, 3]).type(torch.long)
        samples[:, :input_seq, :] = start

        for i in range(num_samples):

            if input_seq == 1:
                hidden = self.init_hidden()
            else:
                hidden = self.init_hidden()
                for j in range(input_seq):
                    _, _, _, hidden = self.forward(samples[i][j], hidden, torch.tensor(j))

            t = input_seq

            while t < 10:
                pre = samples[i][t - 1][1]
                loc, arr, dur, hidden = self.forward(samples[i][t - 1], hidden, torch.tensor(t))
                mid = gen_single(loc, arr, dur, samples[i][t - 1][1] + samples[i][t - 1][2], samples[i][t - 1][0])

                if pre > mid[1]:
                    break
                samples[i][t] = mid

                if mid[1] + mid[2] > 22:
                    break

                t += 1

        return samples

    def pretrain_GAN_single(self, dis, start):

        num_len = start.size()[0]
        loss = 0
        for i in range(num_len):

            hidden = self.init_hidden()
            mid = start[i]
            t = 0
            while True:
                pre = mid[1]
                loc, arr, dur, hidden = self.forward(mid, hidden, torch.tensor(t))
                mid = gen_single(loc, arr, dur, mid[1] + mid[2], mid[0])
                if mid[1] < pre:
                    break
                if mid[1] + mid[2] > 22:
                    break
                loss -= (loc[mid[0]] + arr[mid[1]] + dur[mid[2]]) * dis(mid)
                t += 1

        return loss / num_len

    # def train_MLE(self, inp, target, sample_num):
    #
    #     batch_size = inp.size()[0]
    #     loss_fn = nn.MSELoss()
    #     loss = 0
    #     for i in range(batch_size):
    #         j = 0
    #         hidden = self.init_hidden()
    #         while j < len(target[i]) and target[i][j][0] != 0:
    #             next_loc, next_arr, next_dur, hidden = self.forward(inp[i][j], hidden, torch.tensor(j))
    #             for num in range(sample_num):
    #                 point = gen_single(next_loc, next_arr, next_dur, inp[i][j][1] + inp[i][j][2], inp[i][j][0])
    #                 loc_dis = [torch.abs(point[0] % 50 - target[i][j][0] % 50) + torch.abs(
    #                     torch.floor(point[0] / 50) - torch.floor(point[0] / 50))] * next_loc[point[0]]


# model = MyModel()
# hidden = model.init_hidden()
# x = torch.tensor([1024, 10, 5])
# print(model(x, hidden, torch.tensor(0)))
