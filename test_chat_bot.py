"""
测试chatbot相关api
"""

from chatbot.word2sequence import Word2Sequence
import config
import pickle
from chatbot.dataset import dataloader
from chatbot.train import train
from chatbot.eval import eval
from chatbot.decoder import Beam
from chatbot.chatbot import ChatBot


def save_ws():
    ws = Word2Sequence()
    for line in open(config.chat_input_byword_path, encoding="utf-8").readlines():
        ws.fit(line.strip().split())
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open(config.chat_ws_input_byword_path, "wb"))


    ws = Word2Sequence()
    for line in open(config.chat_target_byword_path, encoding="utf-8").readlines():
        ws.fit(line.strip().split())
    ws.build_vocab()
    print(len(ws))
    # pickle.dump(ws, open(config.chat_ws_input_path, "wb", encoding="utf-8"))
    pickle.dump(ws, open(config.chat_ws_target_byword_path, "wb"))
    # pickle.dump(ws, open(config.chat_ws_target_path, "wb", encoding="utf-8"))


def test_dataloader():
    print(dataloader)
    for idx, (input, target, input_length, target_length) in enumerate(dataloader):
        print(idx)
        print(input, input.cpu().size())
        print(target, target.cpu().size())
        print(input_length, input_length.cpu().size())
        print(target_length, target_length.cpu().size())
        break

def train_seq2seq():
    for i in range(10):
        train(i)


def test_beam():
    heap = Beam()
    heap.add(1, True, ["s", "sdf", "沙发"], "s", "ff")
    heap.add(3, False, ["s", "sdf", "沙发"], "w", "ww")
    heap.add(1, False, ["s", "sdf", "沙发"], "s", "ss")
    heap.add(4, False, ["s", "sdf", "沙发"], "s", "ff")
    # print(heap.heap)
    for i in heap:
        print(i)



if __name__ == '__main__':
    # save_ws()
    # test_dataloader()
    # train_seq2seq()
    # eval()
    # [probility, complete, seq, decoder_input,decoder_hidden]
    # test_beam()
    chatbot = ChatBot()
    ret = chatbot.predict(input("请输入："))
    print(ret)
