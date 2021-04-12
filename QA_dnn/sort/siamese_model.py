"""
构建孪生神经网络：
1.embedding
2.GRU
3.attention
4.attention与 GRU output 进行 concat
5.GRU
6.pooling
7.DNN
"""

import torch.nn as nn
import config
import torch.nn.functional as F
import torch


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.sort_ws),
                                      embedding_dim=config.siamese_embedding_dim,
                                      padding_idx=config.sort_ws.PAD)
        self.gru1 = nn.GRU(input_size=config.siamese_embedding_dim,
                           hidden_size=config.siamese_hidden_size,
                           num_layers=config.siamese_num_layer, batch_first=True,
                           bidirectional=config.bidirectional)
        self.gru2 = nn.GRU(input_size=config.siamese_hidden_size*2*2,
                           hidden_size=config.siamese_hidden_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(config.siamese_hidden_size*4),
            nn.Linear(config.siamese_hidden_size*4, config.siamese_hidden_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.siamese_hidden_size),
            nn.Dropout(config.siamese_drop_out),

            nn.Linear(config.siamese_hidden_size, config.siamese_hidden_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.siamese_hidden_size),
            nn.Dropout(config.siamese_drop_out),

            nn.Linear(config.siamese_hidden_size, 2)
        )

    def forward(self, input, similar_input):
        # 这里使用mask，在后面计算attention的时候，让其忽略pad的位置，等于PAD的位置都为1，其余位置都为0
        # mask1, mask2:[batch_size, max_len]
        mask1, mask2 = input.eq(config.sort_ws.PAD), similar_input.eq(config.sort_ws.PAD)

        # 1.embedding
        input = self.embedding(input)  # [batch_size, max_len , embedding_dim]
        similar_input = self.embedding(similar_input)

        # 2.GRU
        # output: [batch_size, max_len, hidden_size,*num_layer]
        # hidden_state: [num_layer*2, batch_size, hidden_size]
        output, hidden_state = self.gru1(input)
        similar_output, similar_hidden_state = self.gru1(similar_input)

        # 3.attention
        output_align, similar_output_align = self.soft_attention_align(output, similar_output, mask1, mask2)

        # 4.attention与 GRU output 进行 concat
        output = torch.cat([output, output_align], dim=-1)  # [batch_size, max_len , hidden_size * num_layer * 2]
        similar_output = torch.cat([similar_output, similar_output_align], dim=-1)

        # 5.GRU
        gru2_output, _ = self.gru2(output)  # [batch_size, max_len, hidden_size*1]
        gru2_similar_output, _ = self.gru2(similar_output)  # [batch_size, max_len, hidden_size*1]

        # 6.pooling
        output_pool = self.apply_pooling(gru2_output)  # [batch_size, hidden_size*2]
        similar_output_pool = self.apply_pooling(gru2_similar_output)  # [batch_size, hidden_size*2]

        # 7.Concate合并到一起，送入DNN
        out = torch.cat([output_pool, similar_output_pool], dim=-1)  # [batch_size, hidden_size*4]
        out = self.dnn(out)  # [batch_size, 2]

        # return F.log_softmax(out, dim=-1)
        return out

    def soft_attention_align(self, x1, x2, mask1, mask2):
        """self attention
        align 表示对齐的意思
        x1,x2 :[batch_size, max_len, hidden_size,*num_layer]
        x1,x2分别作为encoder和decoder
        """
        # 向PAD的位置（PAD的位置值为1），填充inf
        mask1 = mask1.float().masked_fill(mask1, float("-inf"))
        mask2 = mask2.float().masked_fill(mask2, float("-inf"))
        # 1. 计算 attention weight
        # 2. 计算 attention weight * encoder_output
        _weight = x1.bmm(x2.transpose(1, 2))  # [batch_size, x1_max_len, x2_max_len]
        x1_attention_weight = F.softmax(_weight + mask2.unsqueeze(1), dim=-1)
        x1_align = x1_attention_weight.bmm(x2)

        x2_attention_weight = F.softmax(_weight.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = x2_attention_weight.bmm(x1)

        return x1_align, x2_align

    def apply_pooling(self, input):
        avg_pool = F.avg_pool1d(input.transpose(1, 2), kernel_size=input.size(1)).squeeze(-1)
        max_pool = F.max_pool1d(input.transpose(1, 2), kernel_size=input.size(1)).squeeze(-1)

        return torch.cat([avg_pool, max_pool], dim=-1)  # [batch_size, hidden_size*2]



