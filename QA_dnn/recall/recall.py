"""
返回召回的结果
"""
from QA_dnn.recall.sentence2vector import Sentence2Vector
import config

# 1.搜索句子向量
# 2. 返回结果

class Recall(object):

    def __init__(self, by_word=False, method="FastText"):
        self.by_word = by_word
        self.method =method
        self.sentence_vec = Sentence2Vector(method, byword=self.by_word)
        self.vectorizer, self.features_vec, self.lines_cuted, self.search_index = self.sentence_vec.build_vector()

    def predict(self, sentence):
        """
        仅进行测试使用
        :param sentence:{“q_cut_byword“: str, "q_cut": str, "entity": [str, str...]}
        :return:
        """
        cur_sentence = [sentence["q_cut"]]
        cur_sentence_vector = self.vectorizer.transform(cur_sentence)
        search_result = self.search_index.search(cur_sentence_vector, k=config.recall_topk, k_clusters=config.recall_clusters, return_distance=True)

        # 过滤主体
        # search_result:[ [ (distance1， result1),(distance2， result2),(distance3， result3)... ] ]
        filter_result = []
        for result in search_result[0]:  # result: [(distance1， result1), (distance2， result2), (distance3， result3)...]
            # distance = result[0]
            key = result[1]
            entities = self.sentence_vec.qa_dict[key]["entity"]
            if len(set(entities) & set(sentence["entity"])) > 0:  # 命名体的集合存在交集的时候返回
                filter_result.append(key)

        # 最终返回
        if len(filter_result) < 1:
            return [i[1] for i in search_result[0]]
        else:
            return filter_result