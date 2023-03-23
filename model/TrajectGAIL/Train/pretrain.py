import torch
# import torch.nn as nn
# from model.GAIL import GAIL, Discriminator


# 计算给定state和action的情况下返回的value
# 具体的过程就是给定一个state以及action，先算出一步的reward，注意这一步是不带衰减系数的
# 再从这个状态出发，采样，每一次采样一步都让你的dis去算算这一步的reward是多少,再乘以衰减系数GAMA
# 注意的是，算这个value的目的是让你的值估计器去拟合最终的结果，就是dis_out，用的是MSELoss

def get_value(dis, gen, s, action, seq_len, gama):
    n = len(s[0])
    dis_h = dis.state_emb.init_hidden(batch_size=s.size()[0])
    for i in range(s.size()[1]):
        dis_out, dis_h = dis.gen_state(s[:, i], dis_h)
    reward = dis(dis_out, action)
    dis_out, dis_h = dis.gen_state(action, dis_h)
    gen_h = gen.state_emb.init_hidden(batch_size=s.size()[0])
    s = torch.cat([s, action.view(-1, 1)], dim=1)
    for i in range(s.size()[1]):
        gen_out, gen_h = gen.gen_state(s[:, i], gen_h)
    for i in range(seq_len - len(s[0])):
        state = gen.gen_policy(gen_out)
        action = torch.multinomial(torch.exp(state), num_samples=1)
        dis_out, dis_h = dis.gen_state(action, dis_h)
        reward += dis(dis_out, action) * gama ** i
        s = torch.cat([s, action], dim=1)
        gen_out, gen_h = gen.gen_state(action, gen_h)
    return reward / (seq_len - n)


def train_policy(gen, inp):
    hidden = gen.state_emb.init_hidden(batch_size=inp.size()[0])
    e_loss = 0
    J_loss = 0
    for i in range(inp.size()[1]):
        out, hidden = gen.gen_state(inp[:, i], hidden)
        action_prob = gen.gen_policy(out.detach())
        x = torch.distributions.Categorical(torch.exp(action_prob))
        e_loss += torch.mean(x.entropy())
        action = torch.multinomial(torch.exp(action_prob), num_samples=1)
        value = gen.get_value(out.detach(), action.view(-1))

        value = torch.tensor([action_prob[i][x] for (i, x) in enumerate(action)]) * value
        J_loss += torch.mean(value)
    return -(J_loss + 0.1 * e_loss)


