"""
chatbot的封装
"""
from chatbot.seq2seq import Seq2Seq
import config
import torch
import numpy as np
from lib.cut_sentence import cut


class ChatBot(object):

    def __init__(self):
        self.seq2seq = Seq2Seq().to(config.device)
        self.seq2seq.load_state_dict(torch.load(config.chatbot_seq2seq_byword_model_save_path))
        # print(config.chatbot_ws_input.dict)

    def predict(self, sentence, by_word="cut_byword"):
        """
        :param sentence:  dict:{"cut_byword": str, "cut":str}
        :return:
        """
        cuted_byword = cut(sentence, by_word=True)
        cuted = cut(sentence, by_word=False)
        sentence = {"cut_byword": cuted_byword,
                    "cut": cuted}
        _input = sentence[by_word]
        input_length = torch.LongTensor(
            [len(_input) if len(_input) < config.chatbot_byword_max_len else config.chatbot_byword_max_len]).to(
            config.device)
        _input = torch.LongTensor(
            [config.chatbot_ws_input.transform(_input, max_len=config.chatbot_byword_max_len)]).to(config.device)

        # indices:[[1],[2],[3],...]
        indices = np.array(self.seq2seq.evaluate_beam_search(_input, input_length)).flatten()  # [1,2,3...]
        # print(indices)
        output = "".join(config.chatbot_ws_target.inverse_transform(indices))
        output = output.replace("UNK", "")
        # print("answer:", output)
        return output

if __name__ == '__main__':
    pass