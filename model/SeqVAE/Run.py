import VAE
import DataLoader
import Train
import pandas as pd

df = pd.read_csv("/mnt/data/gonghaofeng/deeplearning_project/ODseq_GAN_remote/data/CDR/HaiKou_Seq.csv")
data = DataLoader.prepare(df)

model = VAE.Model(point_size=2500, emb_dim=32, hidden_dim=32, latent_dim=32)
crite = VAE.VAELoss(batch_size=128)
Train.pretrain(model=model, data=data, seqlen=24, batch_size=128, Epoch=25, lr=1e-2, crite=crite)
