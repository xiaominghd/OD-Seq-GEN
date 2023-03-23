import torch
import VAE
import pandas as pd
import DataLoader

MODEL_FILE = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Result/SVAE/SVAE_10.pth"
GEN_FILE = "/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/Gen_data/SVAE/SVAE_gen.txt"
model = VAE.Model(point_size=2500, emb_dim=32, hidden_dim=32, latent_dim=32)
model.load_state_dict(torch.load(MODEL_FILE))
df = pd.read_csv("/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR/HaiKou_Seq.csv")
data = DataLoader.prepare(df)
#
# with open(GEN_FILE, 'w') as file_obj:
#     for i in range(2000):
#         dec_out, _, _, = model(data[i])
#         for j in range(5):
#             seq = torch.multinomial(torch.exp(dec_out).view(24, -1), num_samples=1).view(-1).numpy().tolist()
#             write_s = ""
#             for s in seq:
#                 write_s = write_s + str(s) + " "
#             write_s += '\n'
#             file_obj.write(write_s)


