"""
模型训练
"""
from QA_dnn.sort.dataset import sort_data_loader
from tqdm import tqdm
from QA_dnn.sort.siamese_model import SiameseNet
import config
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
model = SiameseNet().to(config.device)
optimizer = Adam(model.parameters())


def train(epoch):
    for i in range(epoch):
        bar = tqdm(enumerate(sort_data_loader), total=len(sort_data_loader))
        loss_list = []

        for idx, (_input, similar_input, target) in bar:
            optimizer.zero_grad()
            output = model(_input, similar_input)
            # loss = F.nll_loss(output, target)
            loss = criterion(output, target)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            bar.set_description("epoch:{}\tidx:{}\t{:.6f}".format(i+1, idx, np.mean(loss_list)))

            if idx % 100 == 0:
                torch.save(model.state_dict(), config.sort_model_path)



