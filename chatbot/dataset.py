"""
完成数据集的准备
"""
from torch.utils.data import Dataset, DataLoader
import config
import torch


class ChatBotDataset(Dataset):
    def __init__(self, by_word=True):
        if by_word:
            self.target_path = config.chat_target_byword_path
            self.input_path = config.chat_input_byword_path
        else:
            self.target_path = config.chat_target_path
            self.input_path = config.chat_input_path
        self.input_lines = open(self.input_path, encoding="utf-8").readlines()
        self.target_lines = open(self.target_path, encoding="utf-8").readlines()
        assert len(self.input_lines) == len(self.target_lines), "input和target长度不一致"

    def __getitem__(self, index):
        input = self.input_lines[index].strip().split()
        target = self.target_lines[index].strip().split()
        input_length = len(input) if len(input) < config.chatbot_byword_max_len else config.chatbot_byword_max_len
        target_length = len(target) if len(target) < config.chatbot_byword_max_len else config.chatbot_byword_max_len + 1  # add_eos + 1
        return input, target, input_length, target_length

    def __len__(self):
        return len(self.input_lines)


def collate_fn(batch):
    """

    :param batch: [(input, target, input_length, target_length),(input, target, input_length, target_length),()....]
    :return:
    """
    # 排序
    batch = sorted(batch,key=lambda x:x[2], reverse=True)
    input, target, input_length, target_length = zip(*batch)  # 对应位置合并到一起

    input = [config.chatbot_ws_input.transform(i, config.chatbot_byword_max_len) for i in input]
    target = [config.chatbot_ws_target.transform(i, config.chatbot_byword_max_len, add_eos=True) for i in target]
    input = torch.LongTensor(input).to(config.device)
    input_length = torch.LongTensor(input_length).to(config.device)
    target_length = torch.LongTensor(target_length).to(config.device)
    target = torch.LongTensor(target).to(config.device)

    return input, target, input_length, target_length

dataloader = DataLoader(ChatBotDataset(), batch_size=config.chatbot_batch_size,
           shuffle=True,collate_fn=collate_fn)

if __name__ == '__main__':
    print(dataloader)