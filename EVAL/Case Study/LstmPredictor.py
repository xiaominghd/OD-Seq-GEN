import torch
import torch.nn as nn


class t_Lstm(nn.Module):

    def __init__(self, loc_dim, loc_emb, tim_dim, tim_emb, drop_out, hidden):
        super(t_Lstm, self).__init__()
        self.loc_dim = loc_dim
        self.loc_emb = loc_emb
        self.tim_dim = tim_dim
        self.tim_emb = tim_emb
        self.drop_out = drop_out
        self.hidden = hidden
        self.emb_loc = nn.Embedding(self.loc_dim, self.loc_emb)
        self.emb_tim = nn.Embedding(self.tim_dim, self.tim_emb)
        self.fc1 = nn.Linear(
            nn.Linear(self.loc_emb + self.tim_emb, self.loc_emb+self.tim_emb),
            nn.ReLU(),
            nn.Linear(self.loc_emb + self.tim_emb, self.loc_emb+self.tim_emb),
            nn.ReLU(),
        )
        self.LSTM = nn.LSTM(input_size=self.loc_emb+self.tim_emb, hidden_size=self.hidden,
                            num_layers=3, drop_out=drop_out)
        self.LSTM2out = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.loc_dim)
        )
        self.fc2 = nn.Linear(self.hidden, self.loc_dim)

    def forward(self, x):

