import torch
import math
import torch.optim as optim


def pretrain_NLL(model, inp, target, Epoch, lr, batch_size):
    model_file = '/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/SanYa/OD_seqGAN/gen_S_'
    opt = optim.Adam(model.parameters(), lr=lr)

    mid = int(len(inp) * 0.8)
    inp_train, inp_test = inp[:mid], inp[mid:]
    target_train, target_test = target[:mid], inp[mid:]
    batch_num = math.floor(mid / batch_size)

    for epoch in range(Epoch):
        mid_loss = 0
        for i in range(batch_num):
            opt.zero_grad()

            loss = model.train_NLL(inp_train[i * batch_size:i * batch_size + batch_size],
                                   target_train[i * batch_size:i * batch_size + batch_size])

            loss.backward()
            opt.step()
            mid_loss += loss

        print("第{}次训练后".format(epoch))
        print("训练集的loss为：{}".format(mid_loss / batch_num))
        print("测试集的loss为:{}".format(model.train_NLL(inp_test, target_test)))
        print("*-------------------------*")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_file + str(
                epoch) + '.pth')

    print("预训练完成")

    return model


def train_dis(model, inp, target, lr, Epoch, batch_size, model_name):
    model_file = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/SanYa/OD_seqGAN/dis_"
    opt = optim.Adam(model.parameters(), lr=lr)

    inp_train = inp[:int(0.8 * len(inp))]
    inp_test = inp[int(0.8 * len(inp)):]
    target_train = target[:int(0.8 * len(inp))]
    target_test = target[int(0.8 * len(inp)):]

    batch_num = math.floor(len(inp_train) / batch_size)

    for epoch in range(Epoch):

        mid_loss = 0

        for i in range(batch_num):
            opt.zero_grad()

            loss = model.pretrain(inp_train[i * batch_size:i * batch_size + batch_size],
                                  target_train[i * batch_size:i * batch_size + batch_size])

            loss.backward()
            opt.step()
            mid_loss += loss

        print("第{}次训练后".format(epoch + 1))
        print("训练集上的loss为{}".format(mid_loss / batch_num))
        print("测试集上的loss为{}".format(model.pretrain(inp_test, target_test)))
        print("*-------------------------*")
        out = model.reward(inp[0:1000])
        our_pred = torch.where(out > 0.5, 1, 0)
        res = 0
        for i in range(1000):
            if our_pred[i] == target[i][0]:
                res += 1
        print("训练次数为:{}".format(epoch + 1))
        print("loss为{}".format(mid_loss / batch_num))
        print("准确率为{}".format(res / 1000))
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       model_file + model_name + '_{}'.format(epoch) + ".pth")

# def pretrain_GAN_single(gen, dis, gen_Epoch, dis_Epoch, BatchSize, gen_lr, dis_lr):
#     opt1 = optim.Adam(gen.parameters(), lr=gen_lr)
#     opt2 = optim.Adam(dis.parameters(), lr=dis_lr)
#     df2 = pd.read_csv("../data/haikou_8.csv")
#     traject = choice(df2)
#     traject, _ = prepare_pretrain(traject)
#     batch_num = math.floor(len(traject) / BatchSize)
#     for batch in range(batch_num):
#
#         inp = traject[batch * BatchSize:(batch + 1) * BatchSize]
#         mid_loss1 = 0
#         for i in range(gen_Epoch):
#             opt1.zero_grad()
#             loss = gen.pretrain_GAN_single(dis, inp[:, 0, :])
#             loss.backward()
#             opt1.step()
#             mid_loss1 += loss
#         print("第{}次训练后".format(batch + 1))
#         print("生成器的Loss为{}".format(mid_loss1 / gen_Epoch))
#         # for name, p in gen.named_parameters():
#         #     print(name)
#         #     print(p.grad)
#         #     print(p.requires_grad)
#
#         data2csv(gen, inp[:, 0, :].view(BatchSize, -1, 3), "../data/gen_gan_{}.csv".format(batch))
#         df1 = pd.read_csv("../data/gen_gan_{}.csv".format(batch))
#
#         neg = choice_single_dis(df1, BatchSize)
#         pos = choice_single_dis(df2[batch * BatchSize:(batch + 1) * BatchSize], BatchSize)
#         mid_loss2 = 0
#         for i in range(dis_Epoch):
#             opt2.zero_grad()
#             inp, target = prepare_dis(pos, neg)
#             loss = dis.pretrain(inp, target)
#             loss.backward()
#             opt2.step()
#             mid_loss2 += loss
#         print("判别器的Loss为{}".format(mid_loss2 / dis_Epoch))
#         print("************")
#
#         if batch % 5 == 0:
#             torch.save(gen.state_dict(), '../pretrain/gen_D_' + str(
#                 batch) + '.pth')


#
# def pretrain_MonteCarlo(model, traject, Epoch, lr, batch_size, sample_num):
#     opt = optim.Adam(model.parameters(), lr=lr)
#
#     train = traject[:int(len(traject) * 0.8)]
#     test = traject[int(len(traject) * 0.8):]
#
#     batch_num = math.floor(len(train) / batch_size)
#     inp_train, target_train = prepare_pretrain(train)
#     inp_test, target_test = prepare_pretrain(test)
#
#     for epoch in range(Epoch):
#         mid_loss = 0
#
#         for i in range(batch_num):
#             opt.zero_grad()
#
# loss = model.train_MonteCarlo(inp_train[i * batch_size:i * batch_size + batch_size], target_train[i * batch_size:i
# * batch_size + batch_size].type(torch.float32), sample_num)
#
#             loss.backward()
#             opt.step()
#             mid_loss += loss
#         # for name, p in model.named_parameters():
#         #     print(name)
#         #     print(p.grad)
#         #     print(p.requires_grad)
#
#         print("第{}次训练后".format(epoch))
#         print("训练集的loss为：{}".format(mid_loss / batch_num))
#         print("测试集的loss为:{}".format(model.train_MonteCarlo(inp_test, target_test.type(torch.float32), sample_num=4)))
#         print("*-------------------------*")
#
#         if epoch % 10 == 0:
#             torch.save(model.state_dict(), '../pretrain/gen_M_' + str(
#                 epoch) + '.pth')
#
#     print("预训练完成")
#
#     return model
