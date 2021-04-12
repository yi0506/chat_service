"""
使用fasttext获取词向量
"""
import fasttext
import config
import numpy as np


def build_model(by_word=False):
    if not by_word:
        data_path = r"G:\OneDrive\progaram\python\test\pytorch\chat_service\corpus\QA_corpus\q_cut.txt"
        model = fasttext.train_unsupervised(data_path, epoch=20, wordNgrams=2)
        model.save_model(config.fasttext_model_path)
    else:
        data_path = r"G:\OneDrive\progaram\python\test\pytorch\chat_service\corpus\QA_corpus\q_cut_byword.txt"
        model = fasttext.train_unsupervised(data_path, epoch=20, wordNgrams=2)
        model.save_model(config.fasttext_model_byword_path)


def get_model(by_word=False):
    if not by_word:
        return fasttext.load_model(config.fasttext_model_path)
    else:
        return fasttext.load_model(config.fasttext_model_byword_path)


class FastTextVectorizer(object):
    def __init__(self, byword=False):
        self.model = get_model(byword)

    def transform(self, sentence):
        """

        :param sentence: [sentence1, sentence2...]
        :return:
        """

        result = [self.model.get_sentence_vector(i) for i in sentence]
        return np.array(result)

    def fit_transform(self, sentence):
        return self.transform(sentence)