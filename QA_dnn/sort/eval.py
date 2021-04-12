"""
模型评估
"""

from QA_dnn.sort.dataset import sort_data_loader
from tqdm import tqdm
from QA_dnn.sort.siamese_model import SiameseNet
import config
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
model = SiameseNet().to(config.device)
model.load_state_dict(torch.load(config.sort_model_path))
model.eval()  # 置为评估模式，因为模型中有drop_out、BatchNorm,置为评估模式以后，就自动关闭了


def eval(epoch):
    for i in range(epoch):
        bar = tqdm(enumerate(sort_data_loader), total=len(sort_data_loader))
        loss_list = []
        acc_list = []
        for idx, (_input, similar_input, target) in bar:
            output = model(_input, similar_input)
            loss = criterion(output, target)
            loss_list.append(loss.item())

            # 准确率
            pred = torch.max(output, dim=-1)[-1]  # 取最大值的位置
            acc = pred.eq(target).float().mean()
            acc_list.append(acc)

            print(np.mean(loss_list), np.mean(acc_list))

