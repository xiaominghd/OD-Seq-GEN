import torch
import torch.nn as nn
from NewModel import ResNet


def TrajLen(traject):
    i = 0
    while i < 10 and traject[i][0] != 0:
        i += 1
    return min(i, 10)


class single_dis(nn.Module):

    def __init__(self, point_size=2500, loc_emb_dim=16, tim_emb_dim=16, input_dim=5):
        super(single_dis, self).__init__()
        self.emb_loc = nn.Embedding(point_size, loc_emb_dim)
        self.emb_arr = nn.Embedding(24, tim_emb_dim)
        self.emb_dur = nn.Embedding(24, tim_emb_dim)
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
        self.fc4 = nn.Sequential(
            ResNet(input_dim * 3),
            ResNet(input_dim * 3),
            nn.Linear(input_dim * 3, 1)
        )

    def forward(self, x):
        loc = self.fc1(self.emb_loc(x[0]))
        arr = self.fc2(self.emb_arr(x[1]))
        dur = self.fc3(self.emb_dur(x[2]))
        point = torch.concat([loc, arr, dur], dim=0)
        out = torch.sigmoid(self.fc4(point))

        return out

    def pretrain(self, inp, target):
        batch_size = inp.size()[0]
        loss_fn = nn.BCELoss()
        loss = 0
        for i in range(batch_size):
            out = self.forward(inp[i])
            loss += loss_fn(out, target[i])
        return loss / batch_size


class seq_dis(nn.Module):

    def __init__(self, point_size=2500, loc_emb_dim=16, tim_emb_dim=16, input_dim=5, hidden_dim=16):
        super(seq_dis, self).__init__()
        self.emb_loc = nn.Embedding(point_size, loc_emb_dim)
        self.emb_arr = nn.Embedding(24, tim_emb_dim)
        self.emb_dur = nn.Embedding(24, tim_emb_dim)
        self.hidden_dim = hidden_dim
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
        self.fc4 = nn.Linear(input_dim * 3, hidden_dim)
        self.GRU = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True)
        self.GRU2out = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x):

        x = x[:TrajLen(x)]
        loc = self.fc1(self.emb_loc(x[:, 0]))
        arr = self.fc2(self.emb_arr(x[:, 1]))
        dur = self.fc3(self.emb_dur(x[:, 2]))
        point = self.fc4(torch.concat([loc, arr, dur], dim=1)).view(1, -1, self.hidden_dim)
        out, hidden = self.GRU(point)
        s = torch.sigmoid(self.GRU2out(out[:, -1, :].view(2 * self.hidden_dim)))
        return s

    def pretrain(self, inp, target):

        batch_size = inp.size()[0]
        loss_fn = nn.BCELoss()

        loss = 0

        for i in range(batch_size):
            out = self.forward(inp[i])
            loss += loss_fn(out.view(-1), target[i])

        return loss / batch_size

    def reward(self, inp):
        classify = torch.Tensor(inp.size()[0])

        for i in range(len(inp)):
            out = self.forward(inp[i])

            classify[i] = out

        return classify


class class_dis(nn.Module):

    def __init__(self, point_size=2500, loc_emb_dim=16, tim_emb_dim=16, input_dim=5, hidden_dim=16):
        super(class_dis, self).__init__()
        self.emb_loc = nn.Embedding(point_size, loc_emb_dim)
        self.emb_arr = nn.Embedding(24, tim_emb_dim)
        self.emb_dur = nn.Embedding(24, tim_emb_dim)
        self.hidden_dim = hidden_dim
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
        self.fc4 = nn.Linear(input_dim * 3, hidden_dim)
        self.GRU = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True)
        self.GRU2out = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x):
        t_len = TrajLen(x)
        x = x[:t_len]
        loc = self.fc1(self.emb_loc(x[:, 0]))
        arr = self.fc2(self.emb_arr(x[:, 1]))
        dur = self.fc3(self.emb_dur(x[:, 2]))
        point = self.fc4(torch.concat([loc, arr, dur], dim=1)).view(1, -1, self.hidden_dim)
        out, hidden = self.GRU(point)
        s = torch.sigmoid(self.GRU2out(out[:, -1, :].view(2 * self.hidden_dim)))

        return s, out.view(t_len, -1)[:, :self.hidden_dim]

    def pretrain(self, inp, target):

        batch_size = inp.size()[0]
        loss_fn = nn.BCELoss()

        loss = 0

        for i in range(batch_size):
            out, _ = self.forward(inp[i])
            loss += loss_fn(out.view(-1), target[i])

        return loss / batch_size

    def reward(self, inp):
        classify = torch.Tensor(inp.size()[0])

        for i in range(len(inp)):
            out, _ = self.forward(inp[i])

            classify[i] = out

        return classify


class neighbor_dis(nn.Module):

    def __init__(self, poi_emb=None, poi_emb_dim=16, tim_emb_dim=16, latent_dim=8):
        super(neighbor_dis, self).__init__()
        if poi_emb is None:
            self.poi_emb = nn.Embedding(2500, poi_emb_dim)
        else:
            self.poi_emb = poi_emb
        self.fc1 = nn.Sequential(
            ResNet(2 * poi_emb_dim),
            nn.Linear(2 * poi_emb_dim, latent_dim)
        )
        self.tim_emb = nn.Embedding(24, tim_emb_dim)
        self.fc2 = nn.Sequential(
            ResNet(2 * tim_emb_dim),
            nn.Linear(2 * tim_emb_dim, latent_dim)
        )
        self.fc3 = nn.Sequential(
            ResNet(2 * latent_dim),
            nn.Linear(2 * latent_dim, 1)
        )

    def forward(self, x):
        poi = self.fc1(self.poi_emb(x[:, 0]).view(-1))
        tim = self.fc2(self.tim_emb(x[:, 1]).view(-1))
        p = torch.concat([poi, tim], dim=0)
        out = torch.sigmoid(self.fc3(p))
        return out

    def pretrain(self, inp, target):
        batch_size = inp.size()[0]
        loss_fn = nn.BCELoss()
        loss = 0
        for i in range(batch_size):
            out = self.forward(inp[i])
            loss += loss_fn(out.view(-1), target[i])
        return loss / batch_size

    def reward(self, inp):
        classify = torch.Tensor(inp.size()[0])

        for i in range(len(inp)):
            out = self.forward(inp[i])

            classify[i] = out

        return classify
