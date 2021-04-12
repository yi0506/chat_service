"""
测试dnn的api
"""

from QA_dnn.recall.recall import Recall
from QA_dnn.recall.FastText_vectorizer import build_model, get_model
from QA_dnn.sort.word2sequence import save_sort_ws
from QA_dnn.sort.dataset import sort_data_loader
from QA_dnn.sort.train import train


if __name__ == '__main__':
    recall = Recall(method="FastText")  # FastText BM25 Tfidf
    sentence = {"q_cut_byword": "我 是 永 远 爱 你  的",
                "q_cut": "我 是 永远 爱 你 的",
                "entity": ["我", "你"]}
    result = recall.predict(sentence)
    print(result)
    # build_model(True)
    # build_model(False)
    # save_sort_ws()
    # for i, j, k, l in sort_data_loader:
    #     print(i)
    #     print(j)
    #     print(k)
    #     print(l)
    #     break
    # train(epoch=5)