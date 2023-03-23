import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


def TrajLen(traject):
    i = 0
    while traject[i][0] != 0:
        i += 1
    return i


class Encoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(ResNet(hidden_dim), nn.Linear(hidden_dim, latent_dim))

    def forward(self, x):
        return self.fc1(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, total_items):
        super(Decoder, self).__init__()
        self.linear1 = nn.Sequential(ResNet(latent_dim), nn.Linear(latent_dim, hidden_dim))
        self.linear2 = nn.Sequential(ResNet(hidden_dim), nn.Linear(hidden_dim, total_items))
        # nn.init.xavier_normal_(self.linear1.weight)
        # nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Model(nn.Module):
    def __init__(self, point_size=2500, emb_dim=16, hidden_dim=16, latent_dim=16):
        super(Model, self).__init__()
        self.z_log_sigma = None
        self.z_mean = None
        self.point_size = point_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, total_items=point_size)

        self.item_embed = nn.Embedding(point_size, emb_dim)

        self.enc_gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim
                              )
        self.dec_gru = nn.GRU(input_size=latent_dim, hidden_size=latent_dim)
        self.linear1 = nn.Linear(latent_dim, 2 * latent_dim)
        nn.init.xavier_normal_(self.linear1.weight)

        self.tanh = nn.Tanh()

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = self.linear1(h_enc)

        mu = temp_out[:, :self.latent_dim]
        log_sigma = temp_out[:, self.latent_dim:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * torch.autograd.Variable(std_z, requires_grad=False)

    def forward(self, x):
        x = x.view(1, -1)
        in_shape = x.shape  # [bsz x seq_len] = [1 x seq_len]
        x = x.view(-1)  # [seq_len]
        x = self.item_embed(x)  # [seq_len x embed_size]

        x = x.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x embed_size]

        rnn_out, _ = self.enc_gru(x)  # [1 x seq_len x rnn_size]
        rnn_out = rnn_out.view(in_shape[0] * in_shape[1], -1)  # [seq_len x rnn_size]

        enc_out = self.encoder(rnn_out)  # [seq_len x hidden_size]
        sampled_z = self.sample_latent(enc_out)  # [seq_len x latent_size]

        z_dec, _ = self.dec_gru(sampled_z)

        dec_out = self.decoder(z_dec)  # [seq_len x total_items]

        dec_out = dec_out.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x total_items]

        return dec_out, self.z_mean, self.z_log_sigma


class VAELoss(torch.nn.Module):
    def __init__(self, batch_size):
        super(VAELoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, decoder_output, mu_q, logvar_q, y_true_s, anneal):
        # Calculate KL Divergence loss
        kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), -1))

        # Calculate Likelihood
        dec_shape = decoder_output.shape  # [batch_size x seq_len x total_items] = [1 x seq_len x total_items]

        decoder_output = F.log_softmax(decoder_output, -1).view(dec_shape[1], dec_shape[2])
        likelihood = 0
        for i in range(len(decoder_output)):
            likelihood -= decoder_output[i][y_true_s[i]]
        likelihood = likelihood / dec_shape[1]
        #
        #
        # num_ones = float(torch.sum(y_true_s[0, 0]))
        #
        # likelihood = torch.sum(
        #     -1.0 * y_true_s.view(dec_shape[0] * dec_shape[1], -1) * \
        #     decoder_output.view(dec_shape[0] * dec_shape[1], -1)
        # ) / (float(self.batch_size) * num_ones)

        final = (anneal * kld) + likelihood

        return final
