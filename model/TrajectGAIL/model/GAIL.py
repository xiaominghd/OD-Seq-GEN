import torch
import torch.nn as nn


# Policy_net
# MDP过程，采用的是MLP结构，输入当前状态，产生下一时刻的动作空间分布

class Policy_net(nn.Module):

    def __init__(self, state_dim, action_dim, hidden):
        super(Policy_net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.activation = torch.relu
        self.origin_fc4 = nn.Linear(self.hidden, self.action_dim)

    def forward(self, x):  # batch_size * hidden
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        out = self.origin_fc4(x)
        prob = torch.nn.functional.log_softmax(out, dim=1)
        return prob  # batch_size * action_dim


# value_net
# 输入当前的state，以及action，计算得到value

class Value_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super(Value_net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.activation = torch.relu
        self.fc1 = nn.Linear(in_features=self.hidden + self.action_dim, out_features=self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.fc4 = nn.Linear(self.hidden, 1)

    def one_hotify(self, inp, dim):
        one_hot = torch.FloatTensor(inp.size(0), dim).to()
        one_hot.zero_()
        one_hot.scatter_(1, inp.unsqueeze(1).long(), 1)
        return one_hot

    def forward(self, state, action):  # state:batch_size * hidden action:batch_size
        action = self.one_hotify(action, self.action_dim)
        x = torch.cat([state, action], dim=1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        value = self.fc4(x)
        return value  # batch_size


# RNN_state_emb
# 这一部分是模型的关键，输入一个历史序列，输出其对应的状态
# 在生成器和判别器中均有使用

class RNN_state_emb(nn.Module):

    def __init__(self, state_dim=38000, hidden=32, num_layers=3):
        super(RNN_state_emb, self).__init__()
        self.state_dim = state_dim
        self.hidden = hidden
        self.state_emb = nn.Embedding(self.state_dim, self.hidden)
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.activation = torch.relu
        self.num_layers = num_layers

        self.GRU = nn.GRU(input_size=hidden, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        # GRU是要保证batch_first，输入的第一个维度是batch

    def init_hidden(self, batch_size=1):  # 初始化隐状态 num_layers * batch_size * hidden
        return torch.autograd.Variable(torch.zeros([self.num_layers, batch_size, self.hidden]))

    def forward(self, x, hid):  # x:batch_size * 1 hid:num_layers * batch_size * hidden
        # 注意每一个x只输入一个点的位置
        x = self.state_emb(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        out, hid = self.GRU(x, hid)
        return out.view(out.size()[0], -1), hid  # 第一个参数batch_size * hidden,第二个参数num_layers * batch_size * hidden


# 整个GAIL的生成器部分，包括序列状态生成器，值估计器，决策生成器
class GAIL(nn.Module):
    def __init__(self, state_dim=38000, action_dim=38000, hidden=32, num_layers=3):
        super(GAIL, self).__init__()
        self.state_emb = RNN_state_emb(state_dim=state_dim, hidden=hidden, num_layers=num_layers)  # 一个序列emb
        self.policy = Policy_net(state_dim=state_dim, action_dim=action_dim, hidden=hidden)  # 一个决策网络
        self.value = Value_net(state_dim=state_dim, action_dim=action_dim, hidden=hidden)  # 一个值估计器

    def gen_state(self, x, hidden):  # x:batch_size * 1
        # hidde: state_emb初始化的hidden num_layers * batch_size * hidden
        out, hidden = self.state_emb(x.view(-1, 1), hidden)
        return out, hidden

    def gen_policy(self, state):  # state: batch_size * hidden
        return self.policy(state)  # 输出:batch_size * action_dim

    # 给定state预测action
    def pretrain_policy(self, inp, target):  # inp:batch_size * (seq_len-1) target:batch_size * (seq_len-1)
        loss_fn = nn.NLLLoss()  # 采用的是NLLLoss
        loss = 0
        batch_size = inp.size()[0]
        hidden = self.state_emb.init_hidden(batch_size=batch_size)
        for i in range(inp.size()[1]):  # 循环更新隐状态和state
            state, hidden = self.gen_state(inp[:, i], hidden)
            policy = self.gen_policy(state)  # 这里不用detach，因为state并没有循环的参加运算
            loss += loss_fn(policy, target[:, i].view(-1))  # 第i步的state对应第i步的action
        return loss

    def sample(self, inp, seq_len=12):  # 循环生成长度为seq_len序列

        batch_size = inp.size()[0]
        hidden = self.state_emb.init_hidden(batch_size)
        for i in range(len(inp[0])):
            out, hidden = self.gen_state(inp[:, i], hidden)  # 不能确保输入序列的长度均为1，所以要先对状态进行更新

        for i in range(seq_len - len(inp[0])):  # 得到更新的状态之后，计算每一步的action，再将其进行拼接组成新的序列
            state = self.gen_policy(out)
            action = torch.multinomial(torch.exp(state), num_samples=1)
            inp = torch.concat([inp, action], dim=1)
            out, hidden = self.state_emb(action, hidden)

        return inp.numpy().tolist()  # 转化为n维数组的形式进行输出 batch_size * seq_len

    def get_value(self, state, action):  # 给定state以及action，输出值
        return self.value(state, action)


# 判别器
# 判别器的作用是给定了state的情况下，判断对应的action是否为真
# 整个模型的结构式state_emb + value_net+sigmoid

class Discriminator(nn.Module):

    def __init__(self, state_dim=38000, action_dim=38000, hidden=32, num_layers=3):
        super(Discriminator, self).__init__()
        self.state_emb = RNN_state_emb(state_dim=state_dim, hidden=hidden, num_layers=num_layers)
        self.value_net = Value_net(state_dim=state_dim, action_dim=action_dim, hidden=hidden)

    def gen_state(self, x, hidden):  # x:batch_size * 1
        out, hidden = self.state_emb(x.view(-1, 1), hidden)
        return out, hidden

    def forward(self, state, action):  # 给定一个state和action输出其判定为真的概率,state:batch_size * 1,action:batch_size
        value = self.value_net(state, action.view(-1))
        value = torch.sigmoid(value)
        return value

    def pretrain(self, state, action, target):  # target是某一组state-action的标签
        loss_fn = nn.BCELoss()  # 二分类问题，使用BCELoss
        loss = 0
        hidden = self.state_emb.init_hidden(target.size()[0])
        for i in range(state.size()[1]):
            rnn_out, hidden = self.gen_state(state[:, i], hidden)
            prob = self.forward(rnn_out, action[:, i]).view(-1)
            loss += loss_fn(prob, target)
        return loss
