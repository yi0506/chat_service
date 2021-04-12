"""
准备数据集
"""

from torch.utils.data import DataLoader, Dataset
import config
import torch


class DnnsortDataset(Dataset):
    """
    准备两个数据集进行训练，
        1.一个是问题A的数据集
        2.一个是与问题A相似的问题B，C，D组成的数据集
        3、两个数据集长度要一致
    """
    def __init__(self):
        self.q_lines = open(config.sort_q_cut_path, encoding="utf-8").readlines()
        self.similar_q_lines = open(config.sort_q_cut_path, encoding="utf-8").readlines()
        self.target_lines = open(config.sort_q_cut_target_path, encoding="utf-8").readlines()
        assert len(self.q_lines) == len(self.similar_q_lines) == len(self.target_lines), "数据长度不一致"

    def __getitem__(self, idx):
        q = self.q_lines[idx].split()
        similar_q = self.similar_q_lines[idx].split()
        target = float(self.target_lines[idx].strip())
        # len_q = len(q) if len(q) < config.sort_max_len else config.sort_max_len
        # len_similar_q = len(similar_q) if len(similar_q) < config.sort_similar_max_len else config.sort_similar_max_len
        return q, similar_q, target

    def __len__(self):
        return len(self.q_lines )


def collate_fn(batch):
    # 排序
    # batch = sorted(batch, key=lambda x: x[-2], reverse=True)
    q_lines, similar_q_lines, target = zip(*batch)  # 对应位置合并到一起

    q_lines = [config.sort_ws.transform(i, config.sort_max_len) for i in q_lines]
    similar_q_lines = [config.sort_ws.transform(i, config.sort_similar_max_len) for i in similar_q_lines]

    q_lines = torch.LongTensor(q_lines).to(config.device)
    # q_lines_length = torch.LongTensor(q_lines_length).to(config.device)
    # similar_q_lines_length = torch.LongTensor(similar_q_lines_length).to(config.device)
    similar_q_lines = torch.LongTensor(similar_q_lines).to(config.device)

    target = torch.LongTensor(target).to(config.device)

    return q_lines, similar_q_lines, target


sort_data_loader = DataLoader(dataset=DnnsortDataset(), batch_size=config.sort_batch_size,
           shuffle=True, collate_fn=collate_fn)
