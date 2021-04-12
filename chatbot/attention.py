"""
实现attention
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import config


class Attention(nn.Module):
    def __init__(self, method="concat"):
        super(Attention, self).__init__()
        assert method in ["dot", "general", "concat"], "Method Error"
        self.method = method

        if self.method == "general":
            self.Wa_general = nn.Linear(config.chatbot_encoder_hidden_size, config.chatbot_decoder_hidden_size, bias=False)
        elif self.method == "concat":
            self.Wa_concat = nn.Linear(config.chatbot_decoder_hidden_size + config.chatbot_encoder_hidden_size, config.chatbot_decoder_hidden_size, bias=False)
            self.Va = nn.Linear(config.chatbot_decoder_hidden_size, 1 ,bias=False)


    def forward(self, decoder_hidden_state, encoder_output):
        """

        :param decoder_hidden_state:[num_layer, batch_size, decoder_hidden_size]
        :param encoder_output: [batch_size, seq_len, encoder_hidden_size]
        :return:
        """

        # 1.dot:
        if self.method == "dot":
            # 1.如果num_layer > 1,则只取最后一层的hidden_state
            # 2.维度变换 [batch_size, decoder_hidden_size, 1]
            decoder_hidden_state = decoder_hidden_state[-1, :, :].unsqueeze(0).permute(1, 2, 0)

            # 只要encoder_output和decoder_hidden_state设置大小的一致，就可以进行矩阵乘法,
            # 就是保证 encoder_hidden_size和decoder_hidden_size大小相同
            # [batch_size, seq_len, encoder_hidden_size]*[batch_size, decoder_hidden_size, 1] ----> [batch_size, seq_len, 1]
            # torch.bmm(encoder_output, decoder_hidden_state)
            attention = encoder_output.bmm(decoder_hidden_state).squeeze(-1)  # [batch_size, seq_len, 1] ---> [batch_size, seq_len]
            attention_weight = F.softmax(attention, dim=-1)  # [batch_size, seq_len]

        # 2.general:
        elif self.method == "general":
            batch_size = encoder_output.size(0)
            encoder_seq_len = encoder_output.size(1)
            temp = encoder_output.cpu().detach().numpy().reshape((batch_size * encoder_seq_len, -1))  # 改变形状，送入fc层
            # print(encoder_output.dim())
            encoder_output = torch.tensor(temp).to(config.device)
            # print(encoder_output.equal(torch.tensor(encoder_output.cpu().detach().numpy()).to(config.device)))

            encoder_output = self.Wa_general(encoder_output).view((batch_size, encoder_seq_len, -1))  # [batch_size, seq_len, decoder_hidden_size]
            # print(encoder_output.size())
            decoder_hidden_state = decoder_hidden_state[-1, :, :].unsqueeze(0).permute(1, 2, 0)  # [batch_size, decoder_hidden_size, 1]
            # print(decoder_hidden_state.size())
            attention = encoder_output.bmm(decoder_hidden_state).squeeze(-1)  # [batch_size, seq_len, 1] ---> [batch_size, seq_len]
            # print("attention", attention.size())
            attention_weight = F.softmax(attention, dim=-1)  # [batch_size, seq_len]

        # 3.concat:
        elif self.method == "concat":

            decoder_hidden_state = decoder_hidden_state[-1, :, :].unsqueeze(0).permute(1, 0, 2)  # [batch_size, 1, decoder_hidden_size,]
            decoder_hidden_state = decoder_hidden_state.repeat(1, encoder_output.size(1), 1)  # [batch_size, seq_len, decoder_hidden_size]
            # print(decoder_hidden_state.size())
            # print(encoder_output.size())
            concated = torch.cat([decoder_hidden_state, encoder_output], dim=-1)  # [batch_size, seq_len, decoder_hidden_size + encoder_hidden_size]
            # print(concated.size())

            # concated三维变成二维,否则无法送入Linear()
            batch_size = encoder_output.size(0)
            encoder_seq_len = encoder_output.size(1)
            concated = concated.view((batch_size * encoder_seq_len, -1))  # [batch_size * seq_len, decoder_hidden_size + encoder_hidden_size]
            # print(concated.size())
            # print(self.Wa_concat)
            attention = self.Va(torch.tanh(self.Wa_concat(concated))).squeeze(-1)  # [batch_size * seq_len]
            attention_weight = F.softmax(attention.view(batch_size, encoder_seq_len), dim=-1)  # [batch_size, seq_len]

        return attention_weight