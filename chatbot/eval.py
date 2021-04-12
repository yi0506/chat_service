"""
实现模型的评估
"""

from chatbot.dataset import dataloader

from chatbot.seq2seq import Seq2Seq
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import torch
import config
from lib import cut
import os
import numpy as np

"""
    1.实例化model, optimizer, loss
"""
def eval(by_word=True):
    seq2seq = Seq2Seq().to(config.device)
    seq2seq.load_state_dict(torch.load(config.chatbot_seq2seq_byword_model_save_path))
    # print(config.chatbot_ws_input.dict)

    while True:
        _input = input("请输入：")
        _input = cut(_input, by_word=by_word)
        input_length = torch.LongTensor([len(_input) if len(_input) < config.chatbot_byword_max_len else config.chatbot_byword_max_len]).to(config.device)
        _input = torch.LongTensor([config.chatbot_ws_input.transform(_input, max_len=config.chatbot_byword_max_len)]).to(config.device)

        # indices:[[1],[2],[3],...]
        indices = np.array(seq2seq.evaluate_beam_search(_input, input_length)).flatten()  # [1,2,3...]
        # print(indices)
        output = "".join(config.chatbot_ws_target.inverse_transform(indices))
        output = output.replace("UNK", "")
        print("answer:", output)

