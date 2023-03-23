import torch
from NewModel import MyModel
FILE = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/SanYa/OD_seqGAN/gen_GAN_10.pth"
model = MyModel(loc_emb_dim=16, tim_emb_dim=16, pos_emb_dim=32, input_dim=5, point_size=2500, hidden_dim=32)
