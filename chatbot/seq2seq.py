"""
把encoder和decoder进行合并，得到seq2seq模型
"""
import torch.nn as nn
from chatbot.encoder import Encoder
from chatbot.decoder import Decoder
import config


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input, target, input_length, target_length):
        encoder_outputs, encoder_hidden, outputs_length = self.encoder(input, input_length)
        decoder_outputs, decoder_hidden = self.decoder(target, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def evaluate(self, input, input_length):
        encoder_outputs, encoder_hidden, outputs_length = self.encoder(input, input_length)
        indices = self.decoder.evaluate(encoder_hidden, encoder_outputs)
        # print("seq2seq", indices)
        # print(len(indices))
        return indices

    def evaluate_beam_search(self, input, input_length):
        encoder_outputs, encoder_hidden, outputs_length = self.encoder(input, input_length)
        indices = self.decoder.evaluate_beam(encoder_hidden, encoder_outputs)
        return indices
