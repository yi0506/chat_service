"""实现解码器"""

import torch.nn as nn
import torch.nn.functional as F
import config
import torch
import random
from chatbot.attention import Attention
import heapq
from chatbot.word2sequence import Word2Sequence


class Beam(object):
    def __init__(self):
        self.heap = list()
        self.beam_width = config.beam_width

    def add(self, probility, complete, seq, decoder_input, decoder_hidden):
        """
        添加数据，同时判断总的数据个数，多余beam_width，则删除
        :param probility: 概率乘积
        :param complete: 最后一个是否为EOS
        :param seq: 所有token的列表
        :param decoder_input: 下一次进行解码的输入，通过前一次获得
        :param decoder_hidden: 下一次进行解码的hidden，通过前一次获得
        :return:decoder_outputs:[batch_size, seq_len, vocab_size]
        """
        heapq.heappush(self.heap, [probility, complete, seq, decoder_input,decoder_hidden])

        # 判断数据的个数，如果大，则弹出堆，保证个数不多于beam_width个
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):  # 让该heap能够被迭代
        return iter(self.heap)




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.chatbot_ws_target),
                                      embedding_dim=config.chatbot_embedding_dim,
                                      padding_idx=config.chatbot_ws_target.PAD)
        self.gru = nn.GRU(input_size=config.chatbot_embedding_dim,
                          hidden_size=config.chatbot_decoder_hidden_size,
                          num_layers=config.chatbot_decoder_num_layer,
                          batch_first=True)
        self.fc = nn.Linear(config.chatbot_decoder_hidden_size, len(config.chatbot_ws_target))
        self.attention = Attention()
        self.Wa = nn.Linear(config.chatbot_decoder_hidden_size + config.chatbot_encoder_hidden_size, config.chatbot_encoder_hidden_size, bias=False)

    def forward(self, target, encoder_hidden, encoder_outputs):
        """
        :param target:目标值 [batch_size, max_len + 1]
        :param encoder_hidden:[1*1, batch_size, hidden_size]
        :return:
        
        
        1. 使用什么样的损失函数，预测值需要是什么格式的

            - 结合之前的经验，我们可以理解为当前的问题是一个分类的问题，即每次的输出其实是选择一个概率最大的词
            - 预测值真实值的形状是`[batch_size,max_len]`，从而我们知道预测值需要是一个`[batch_size,max_len,vocab_size]`的形状output，**对于每一个句子中的每一个词，是从vacab_size大小的字典中选择概率最大的那个作为输出结果。**
            - 即预测值的最后一个维度进行计算log_softmax,然后和真实值进行相乘，从而得到损失（交叉熵损失）
            

        2. 如何把编码结果`[1,batch_size,hidden_size]`进行操作，得到预测值。可以使用LSTM or GRU的结构，所以在解码器中：
        
            - 通过循环，每次循环输出一个句子中一个单词的预测概率值，当`output=="<END>"`退出循环
            - 第一次循环：编码器的结果作为初始的隐层状态，定义一个`[batch_size,1]`的全为`SOS`的数据作为最开始的输入，告诉解码器，要开始工作了，`[batch_size,1]`就是每次预测值的形状
            - 第二次循环以后：每次循环，LSTM的`input`为上一次输出的预测结果（具体的预测结果，即概率最大值的索引，而不是概率值），`h0`为上一次的`hidden_state`
            - 通过解码器得到输出，输出形状为`[batch_size,hidden_size]`，通过FC层进行形状变换，形状的调整为`[batch_size,vocab_size]`，接着计算softmax交叉熵损失得到预测概率，预测概率形状为`[batch_size,vocab_size]`，再进行概率预测，得到预测结果`[batch_size, 1]`，将预测结果作为下一次解码器输入
            - 上述是一个循环，句子长度是不定的，可以使用max_len进行最大长度限制，防止死循环，不到`<END>`也退出循环
            - 把所有输出的结果进行concate，得到`[batch_size,seq_len,vocab_size]`
        
        
        
        """
        # 1.获取encoder的输出，作为decoder第一次的hidden_state
        decoder_hidden = encoder_hidden
        # print(decoder_hidden.size())

        # 2. 准备decoder第一个时间步的输入，形状为[batch_size,1] 的 SOS 作为输入
        batch_size = encoder_hidden.size(1)
        # batch_size = target.size(0)
        decoder_input = torch.LongTensor(torch.ones(batch_size, 1, dtype=torch.int64) * config.chatbot_ws_target.SOS)\
            .to(config.device)  # [batch_size, 1]

        # 3. 在第一个时间步上进行计算，得到第一个时间步的输出，decoder_output_t, decoder_hidden
        # 4. 把前一个时间步的输出进行预测计算，选出可能性最大的一个，作为第一个最终的输出结果：index
        # 5. 把前一次hidden_state 作为当前时间步的hidden_state的输入，把前一次的输出index，作为当前时间步的输入
        # 6. 循环4-5步骤

        # 保存log_softmax的结果，在模型中用来与target值计算nll_loss损失，梯度下降
        decoder_outputs = torch.zeros([batch_size, config.chatbot_byword_max_len + 1, len(config.chatbot_ws_target)]).to(config.device)

        if random.random() > config.chatbot_teacher_forcing_ratio:  # 加入teacher forcing机制，加快收敛
            for t in range(config.chatbot_byword_max_len + 1):  # +add_eos加1
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)

                # 保存decoder_output_t到decoder_outputs中
                decoder_outputs[:, t, :] = decoder_output_t
                decoder_input = target[:, t].unsqueeze(-1)  # target[:, t]: torch.Size([128]), target[:, t].unsqueeze(-1):[batch_size, 1],总共seq_len个


        else:  # 不使用teacher forcing
            for t in range(config.chatbot_byword_max_len + 1):  # add_eos加1
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)

                # 保存decoder_output_t到decoder_outputs中
                decoder_outputs[:, t, :] = decoder_output_t
                value, index = torch.topk(decoder_output_t, k=1, dim=-1)  # 得到预测的输出index
                # index:[batch_size, 1], 实际上就是num_sequence中序列化以后的indices,我们就是要这个结果，作为下一次时间步输入
                # value：[batch_size, 1], 实际上就是log_softmax计算损失之后的概率值，我们要的不是这个

                decoder_input = index
                # print("*"*20)
                # print("index:",index)
                # print("value:", value)

        return decoder_outputs, decoder_hidden



    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        """
        计算每个时间步上的结果(只有一个时间步)
        :param decoder_hidden: [1*1, batch_size, hidden_size] 也就是编码器encoder_hidden的形状
        :param decoder_input: [batch_size, 1]  "1"表示一个时间步的输入,因为只有一个时间步，所以不需要打包和解包
        """
        decoder_input_embedded = self.embedding(decoder_input)  # decoder_input_embedded: [batch_size, 1, embedding_dim]

        # out: [batch_size, 1, hidden_size]
        # decoder_hidden: [1, batch_size, hidden_size] 与输入的decoder_hidden一致
        out, decoder_hidden = self.gru(decoder_input_embedded, decoder_hidden)
        # print(decoder_hidden.size())


        ###############添加attention ################

        # outputs:[batch_size, input_seq_len, input_hidden_size]
        attention_weight = self.attention(decoder_hidden, encoder_outputs).unsqueeze(1)  # attention_weight:[batch_size, 1, input_seq_len]
        context_vector = attention_weight.bmm(encoder_outputs)  # [batch_size, 1, input_hidden_size]
        concated = torch.cat([out, context_vector], dim=-1)  # [batch_size, 1, decoder_hidden_size + encoder_hidden_size]
        concated = concated.squeeze(1)  # [batch_size, decoder_hidden_size + encoder_hidden_size]
        out = torch.tanh(self.Wa(concated))  # [batch_size, hidden_size]

        ################# attention结束 ##################

        # 处理输出的形状
        # out = out.squeeze(1)  # [batch_size, 1, hidden_size] ---> [batch_size, hidden_size]
        out = self.fc(out)  # [batch_size, vocab_size] vocab_size:字典的大小len(config.num_sequence)

        # 计算损失
        output = F.log_softmax(out, dim=-1)
        # print("output:", output.size())

        return output, decoder_hidden

    def evaluate(self, encoder_hidden, encoder_outputs):
        """
        模型评估
            （1）和encoder大致相同，但是不需要保存每个时间步的output，只需要通过其计算的到预测值
            （2）每个时间步的预测值放在列表中，每一列才是输入的最终结果
        :param encoder_hidden:
        :return: indices:  形状：[seq_len, batch_size]

                            [
                            [batch_size,1]  -->  batch_size个输入的第1个输出
                            [batch_size,1]  -->  batch_size个输入的第2个输出
                            [batch_size,1]  -->  batch_size个输入的第3个输出
                            [batch_size,1]  -->  batch_size个输入的第4个输出
                            ....
                            ]

                            每一列是一条数据的输出

        """
        decoder_hidden = encoder_hidden
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor(torch.ones(batch_size, 1, dtype=torch.int64) * config.chatbot_ws_target.SOS).to(config.device)

        indices = []
        # while True:
        for i in range(config.chatbot_byword_max_len + 5):  # 输出内容的长度，即句子的长度，+10是为了多输出几个词，如果句子很长
            # decoder_output_t:[batch_size,vacab_size]
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            value, index = torch.topk(decoder_output_t, k=1, dim=-1)
            decoder_input = index  # [batch_size, 1]
            # if index == config.num_suquence.EOS:
            #     break
            indices.append(index.squeeze(-1).cpu().detach().numpy())

        return indices


    def evaluate_beam(self, encoder_hidden, encoder_outputs):
        """使用 堆 来完成beam_search，堆是一种优先级的队列，按照优先级顺序存取数据"""

        batch_size = encoder_hidden.size(1)
        # 1.构造第一次需要输入的数据，保存在堆中
        decoder_input = torch.LongTensor([[Word2Sequence.SOS] * batch_size]).to(config.device)  # [batch_size, 1]
        decoder_hidden = encoder_hidden

        prev_beam = Beam()
        prev_beam.add(1, False, [decoder_input], decoder_input, decoder_hidden)

        while True:
            cur_beam = Beam()
            # 2. 取出堆中的数据，进行forward_step操作，获得当前时间步的output、hidden
            # 这里使用下划线进行区分
            for _probility, _complete, _seq, _decoder_input, _decoder_hidden in prev_beam:
                # 判断前一次的 _complete是否为True，如果是，则不需要forward
                # 有可能为True，但是概率不大
                if _complete == True:
                    cur_beam.add(_probility, _complete, _seq, _decoder_input, _decoder_hidden)
                else:
                    decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)

                    value, index = torch.topk(decoder_output_t, k=config.beam_width)  # [batch_size=1, beam_width=3]

                    # 3.从output中选择topk(k=beam_width)个输出，作为下一次的input
                    for m, n in zip(value[0], index[0]):  # value[0]: torch.Size([3])  value: torch.Size([1, 3])
                        decoder_input = torch.LongTensor([[n]]).to(config.device)  # torch.Size([1, 1])
                        seq = _seq + [n]  # 两个列表相加，seq中第一个元素为 tensor([[2]], device='cuda:0')，二维数组，其余元素为一个常数tensor(1608, device='cuda:0')。整体为一个数字序列，需要经过inverse_transform，转化为字符串
                        probility = _probility * m
                        if n.item() == config.chatbot_ws_target.EOS:
                            complete = True
                        else:
                            complete = False

                        # 4. 把下一个时间步需要的输入数据保存在一个新的堆中
                        cur_beam.add(probility, complete, seq, decoder_input, decoder_hidden)
            # 5. 获取新的堆中的优先级最高（概率最大）的数据，判断数据是否为EOS结果，或者达到最大长度，如果是，停止迭代
            # cur_beam里面最多有9个值 3*3
            best_prob, best_complete, best_seq, _, _ = max(cur_beam)
            if best_complete == True or len(best_seq) == config.chatbot_byword_max_len + 1:  # 减去EOS
                return self._prepar_seq(best_seq)

            else:  #6. 否则重新遍历新的堆中的数据,此时cur_beam中有3个数据
                prev_beam = cur_beam


    def _prepar_seq(self, seq):  # 对结果进行基础的处理，共后续转化为文字使用
        if seq[0].item() == config.chatbot_ws_target.SOS:
            seq = seq[1:]
        if seq[-1].item() == config.chatbot_ws_target.EOS:
            seq = seq[:-1]
        seq = [i.item() for i in seq]
        return seq




if __name__ == '__main__':
    pass