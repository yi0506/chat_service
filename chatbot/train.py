from chatbot.dataset import dataloader

from chatbot.seq2seq import Seq2Seq
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import torch
import config
import torch.nn as nn
import os

"""
训练流程:
    1.实例化model, optimizer, loss
    2.遍历dataloader
    3.调用模型得到output
    4.计算损失
    5.模型加载保存
"""

seq2seq = Seq2Seq().to(config.device)
optimizer = Adam(seq2seq.parameters(), lr=0.001)


def train(epoch):
    init_loss = 10
    loss_list = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="train")
    for index, (input, target, input_length, target_length) in bar:
        input = input.to(config.device)
        target = target.to(config.device)
        input_length = input_length.to(config.device)
        target_length = target_length.to(config.device)

        optimizer.zero_grad()
        decoder_outputs, _ = seq2seq(input, target, input_length, target_length)
        # print(decoder_outputs.size(), target.size())
        # decoder_outputs转换为二维，target转换为一维的形状，nll_loss在三维的时候计算损失有问题
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(0) * decoder_outputs.size(1),
                                               -1)  # [batch_size*seq_len, vocab_size]
        target = target.view(-1)
        # print(target)
        # print(decoder_outputs)
        # print("转换后", decoder_outputs.size(), target.size())
        loss = F.nll_loss(decoder_outputs, target, ignore_index=config.chatbot_ws_target.PAD)
        loss.backward()

        # 进行梯度裁剪
        nn.utils.clip_grad_norm(seq2seq.parameters(), config.clip)

        optimizer.step()
        bar.set_description("epoch:{}\tindex:{}\tloss:{:.3f}".format(epoch + 1, index, loss.item()))

        if epoch - 7 <= 0:  # 前7次epoch，每100次参数更新保存一次模型
            if index % 100 == 0:
                torch.save(seq2seq.state_dict(), config.chatbot_seq2seq_byword_model_save_path)
                # torch.save(optimizer.state_dict(), config.chatbot_seq2seq_byword_optimizer_save_path)

        if epoch - 7 > 0:  # 最后2次epoch，只保存比上一次损失小的模型
            if init_loss > loss.item():
                # print(loss.item())
                init_loss = loss.item()
                # print(init_loss)
                loss_list.append(init_loss)

                # 保存模型
                torch.save(seq2seq.state_dict(), config.chatbot_seq2seq_byword_model_save_path)
                # torch.save(optimizer.state_dict(), config.chatbot_seq2seq_byword_optimizer_save_path)
                # print("save success")
        # if init_loss > loss.item():
        #     init_loss = loss.item()
        #     loss_list.append(init_loss)
        #     torch.save(seq2seq.state_dict(), config.chatbot_seq2seq_byword_model_save_path)
        #     torch.save(optimizer.state_dict(), config.chatbot_seq2seq_byword_optimizer_save_path)

        # if index == len(dataloader) - 1 and len(loss_list) > 0:  # 对最后几轮最好的模型改名，加上损失
        #     # print(len(loss_list))
        #     # assert len(loss_list) > 0, "loss_list为空"
        #     min_loss = min(loss_list)
        #     # print(min_loss)
        #     # print("rename success")
        #     os.rename(config.chatbot_seq2seq_byword_model_save_path,
        #               config.chatbot_seq2seq_byword_model_save_path + "loss:{:.3f}".format(min_loss))
            # print("rename success")


if __name__ == '__main__':
    for i in range(10):
        train(i)
