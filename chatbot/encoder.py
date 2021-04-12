"""
编码器
"""
from torch import nn
import config
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.chatbot_ws_input),
                                      embedding_dim=config.chatbot_embedding_dim,
                                      padding_idx=config.chatbot_ws_input.PAD)
        self.gru = nn.GRU(input_size=config.chatbot_embedding_dim,
                          num_layers=config.chatbot_encoder_num_layer,
                          hidden_size=config.chatbot_encoder_hidden_size,
                          batch_first=True)


    def forward(self, input, input_length):
        """
        pack_padded_sequence 使用过程中需要对batch中的内容按照句子的长度 "降序排序"
        :param input:[batch_size, max_len]
        :return:
        """
        embeded = self.embedding(input)  # embeded : [batch_size, embedding_dim]

        # 打包
        embeded = pack_padded_sequence(embeded, lengths=input_length, batch_first=True)

        out, encoder_hidden = self.gru(embeded)

        # 解包
        out, out_length = pad_packed_sequence(out, batch_first=True,
                                              padding_value=config.chatbot_ws_input.PAD,
                                              total_length=None)  # 不指定的话，total_length默认为句子的最大长度，使 out 的 seq_len 为最长句子的长度

        # encoder_hidden:[1*1, batch_size, hidden_size]
        # out: [batch_size, seq_len, hidden_size]
        return out, encoder_hidden, out_length


if __name__ == '__main__':
    from dataset import train_data_loader
    encoder = Encoder()
    print(encoder)
    for input, target, input_length, target_length in train_data_loader:
        out, encoder_hidden, out_length = encoder(input, input_length)
        print(out.size())
        print(encoder_hidden.size())
        print(out_length)
        break
