"""
意图识别模型封装
"""
import config
import fasttext

class Classify():
    def __init__(self):
        """
        加载训练好的模型
        """
        self.model_byword = fasttext.load_model(config.final_model__byword_path)  # 单个字的模型
        self.model = fasttext.load_model(config.final_model_path)  # 按照词语的模型

        pass

    def predict(self, sentence):
        """
        预测输入数据的结果，并返回准确率
        :param sentence: {"cut_byword": str, "cut": str}
        :return:(label, acc)
        """
        result1 = self.model.predict(sentence["cut"])
        result2 = self.model_byword.predict(sentence["cut_byword"])
        for label, acc, label_byword, acc_byword in zip(*result1, *result2):
        # for label, acc, label_byword, acc_byword in zip(label, acc, label_byword, acc_byword):

            # 当两个模型预测的label不一致时（概率很小很小），
            # 我们不能比较两个不同label的概率值，需要转换到同一个label类别上，去比较两个模型预测的概率
            # 在同一个lable类别上，哪个概率大，就使用哪个模型的预测结果
            # 所以，都转换到chat这个类别的概率
            if label == "__label__chat":
                label = "__label__QA"
                acc = 1 - acc
            if label_byword == "__label__chat":
                label_byword = "__label__QA"
                acc_byword = 1 - acc_byword

            # 判断准确率
            if acc > 0.96 or acc_byword > 0.96:  # 设置阈值
                return ("QA", max(acc, acc_byword))

            else:  # 当两个模型都预测为chat时，由于进行了类别转换，chat类别的概率很小，chat概率越小，说明原本概率越大
                return ("chat", 1 - min(acc, acc_byword))

            # TODO 假设有三个类别,上面的方法只适用于二分类
            # if label == label_byword:
            #     if acc > 0.95 or acc_byword > 0.95:
            #         return label, max(acc, acc_byword)
            #     else:
            #         return None, 0  # 无法获取分类，或是不符合阈值要求
            #
            # elif acc_byword > 0.99:  # 返回单个字的模型结果
            #     return  label_byword, acc_byword
            # elif acc > 0.98:  # 返回词语模型的结果
            #     return label, acc
            #
            # else:
            #     return None, 0  # 无法获取分类，或是不符合阈值要求
