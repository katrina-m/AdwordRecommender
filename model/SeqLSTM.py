import torch
from utility.dnn_components import PointWiseFeedForward, MultiHeadAttention, TimeEncode, PosEncode, EmptyEncode, \
    MyMeanPool
from model.BaseModel import BaseModel
import numpy as np


class SeqLSTM(BaseModel):

    def __init__(self, args):
        super(SeqLSTM, self).__init__(args)

        if self.use_time == 'time':
            self.time_encoder = TimeEncode(time_dim=self.hidden_units)
        elif self.use_time == 'pos':
            self.time_encoder = PosEncode(time_dim=self.hidden_units, seq_len=self.time_span)
        elif self.use_time == 'empty':
            self.time_encoder = EmptyEncode(time_dim=self.hidden_units)
        else:
            raise ValueError('invalid time option!')

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        # Multi-head attention layers
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = MultiHeadAttention(self.num_heads, self.hidden_units, \
                                                self.hidden_units, self.hidden_units, dropout=self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        # Auxiliary features
        self.cat_emb = torch.nn.Embedding(10, 3)
        self.sub_cat_emb = torch.nn.Embedding(56, 8)
        self.month_day_emb = torch.nn.Embedding(31, 5)
        self.month_emb = torch.nn.Embedding(12, 3)
        self.year_emb = torch.nn.Embedding(10, 3)
        self.holiday_emb = torch.nn.Embedding(15, 3)
        self.keyword_emb = torch.nn.Embedding(140133, 32)
        self.meanpool = MyMeanPool(axis=1)

        self.seq_project_layer = torch.nn.Linear(1, self.hidden_units)
        self.input_project_layer = torch.nn.Linear(32, self.hidden_units)
        self.query_project_layer = torch.nn.Linear(11, self.hidden_units)
        self.output_layer = torch.nn.Linear(32, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, input):
        seqs, isnull_X, month_day, month, year, holiday, timestamp, categories, sub_categories, rootwords = input

        # Board features
        categories_emb = self.cat_emb(categories)
        sub_categories_emb = self.sub_cat_emb(sub_categories)

        # Time embedding
        month_day_emb = self.month_day_emb(month_day)
        month_emb = self.month_emb(month)
        year_emb = self.year_emb(year)
        holiday_emb = self.holiday_emb(year)
        time_emb = torch.cat([month_day_emb, month_emb, year_emb, holiday_emb], axis=2)
        time_feat = torch.cat([month_day.unsqueeze(-1), month.unsqueeze(-1), year.unsqueeze(-1), holiday.unsqueeze(-1)], axis=2)

        # Apply mean pooling for root-word representation
        rootwords_emb = self.keyword_emb(rootwords)
        rootword_mask = ~(rootwords == 0)
        rootword_mask = rootword_mask.type(torch.FloatTensor).to(self.device)
        rootwords_emb = self.meanpool(rootwords_emb, rootword_mask)

        seqs = self.seq_project_layer(seqs)
        #seqs = torch.cat([time_emb[:, :self.time_span, :], seqs], axis=2)
        #seqs = self.input_project_layer(seqs)
        seqs = torch.cat([seqs, self.time_encoder(timestamp[:, :self.time_span])], axis=2)
        #seqs = torch.cat([seqs, time_feat[:, :self.time_span, :].type(torch.FloatTensor).to(self.device)], axis=2)
        seqs = self.input_project_layer(seqs)
        seqs = self.dropout(seqs)

        timeline_mask = isnull_X

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        auxiliary_feat = torch.cat([categories_emb, sub_categories_emb], axis=1)
        query_feat = self.query_project_layer(auxiliary_feat)
        for i in range(len(self.attention_layers)):
            # if i == len(self.attention_layers)-1: # Last layer use normal attention
            #     Q = self.attention_layernorms[i](auxiliary_feat.unsqueeze(1))
            #     mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=None, mask=timeline_mask)
            # else:
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask, mask=timeline_mask)
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        final_feat = mha_outputs[:, -1, :]

        final_feat = torch.cat([final_feat, query_feat], axis=1)
        logits = self.output_layer(final_feat)
        return logits

    # def forward_v0(self, input):
    #     X, isnull_X = input
    #     sequence_input = torch.cat([X, isnull_X.unsqueeze(-1)], 2)
    #     seqs, _ = self.gru_layer(sequence_input)  # (batch_size, seq_len, hidden_dim)
    #     seqs = self.forward_layer(seqs)
    #     final_feat = seqs[:, -1, :]  # only use last QKV classifier, a waste
    #     logits = self.output_layer(final_feat)
    #     return logits

    def predict(self, input):
        return self.forward(input)

    def update_loss(self, optimizer, batch):
        pos_features, neg_features = batch
        pos_logits = self.forward(pos_features)
        neg_logits = self.forward(neg_features)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(
            neg_logits.shape, device=self.device)
        optimizer.zero_grad()
        loss = self.criterion(pos_logits, pos_labels)
        loss += self.criterion(neg_logits, neg_labels)
        for param in self.keyword_emb.parameters():
             loss += self.l2 * torch.norm(param)
        loss.backward()
        optimizer.step()
        return loss

    def reset_parameters(self):
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass
