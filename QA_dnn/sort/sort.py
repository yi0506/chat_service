"""
排序代码的封装
"""
from QA_dnn.sort.siamese_model import SiameseNet
import config
import torch
import torch.nn as nn
from QA_dnn.recall.sentence2vector import Sentence2Vector


class DnnSort(object):
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.model = SiameseNet().to(config.device)
        self.model.load_state_dict(torch.load(config.sort_model_path))

        self.model.eval()  # 置为评估模式，因为模型中有drop_out、BatchNorm,置为评估模式以后，就自动关闭了
        self.sentence2vector = Sentence2Vector()

    def predict(self, sentence, recall_list):
        """

        :param sentence: {"q_cut_byword": srt,
                            "q_cut": "str,
                            "entity": [sg1, sg2]}
        :param recall_list:
        :return:
        """
        input = [sentence["q_cut"]] * len(recall_list)  # 用户输入的句子
        similar_input = [self.sentence2vector.qa_dict[i]["q_cut"] for i in recall_list]  # 召回的结果
        _input = torch.LongTensor([config.sort_ws.transform(i, config.sort_max_len) for i in input]).to(config.device)
        _similar_input = torch.LongTensor([config.sort_ws.transform(i, config.sort_similar_max_len) for i in similar_input]).to(config.device)

        output = self.model(_input, _similar_input)
        output = self.criterion(output)  # [batch_size, 2] 相似与不相似的概率
        output = output[:, -1].squeeze(-1).detach().cpu().numpy()
        best_question, best_prob = sorted(list(zip(recall_list, output)), key=lambda x: x[1], reverse=True)[0]  # 取第一个，最好的结果

        if best_prob > 0.98:
            return self.sentence2vector.qa_dict[best_question]["answer"]
        else:
            return "不懂你在说什么"
