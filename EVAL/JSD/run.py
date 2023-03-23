import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import model.OD_seqGEN.DataLoader as DataLoader
from JSD import JSD_Metrix

Real_file = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR_HaiNan/Sanya/SanYa.csv"
Gen_file = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/SanYa/OD_seqGAN/"


def view(data_len):
    df = pd.read_csv(Real_file)
    traject = DataLoader.choice(df, num=40000)

    duration = []
    traject_len = []
    topk = []
    start = []
    end = []
    distance = []
    radius = []

    for i in range(0, data_len):
        df1 = pd.read_csv(Gen_file + "/gen_gan_{}.csv".format(i))
        df1.columns = ['no', 'id', 'gps2id', 'start_hour', 'duration']
        traject1 = DataLoader.choice(df1, num=8000)
        m = JSD_Metrix(traject, traject1)
        traject_len.append(m.get_JSD_trajlen())
        topk.append(m.get_JSD_Loc())
        duration.append(m.get_JSD_duration())
        start.append(m.get_JSD_start())
        end.append(m.get_JSD_end())
        distance.append(m.get_JSD_distance())
        radius.append(m.get_redius())
    print(distance)
    print(traject_len)
    print(topk)
    print(duration)
    print(start)
    print(radius)

    x = np.linspace(0, data_len, data_len)
    plt.subplot(2, 3, 1)
    plt.title("distance")
    plt.plot(x, distance)
    plt.subplot(2, 3, 2)
    plt.title("traject_len")
    plt.plot(x, traject_len)
    plt.subplot(2, 3, 3)
    plt.title("topK")
    plt.plot(x, topk)
    plt.subplot(2, 3, 4)
    plt.title("duration")
    plt.plot(x, duration)
    plt.subplot(2, 3, 5)
    plt.title("start")
    plt.plot(x, start)
    plt.subplot(2, 3, 6)
    plt.title("radius")
    plt.plot(x, radius)
    plt.show()


view(data_len=31)
