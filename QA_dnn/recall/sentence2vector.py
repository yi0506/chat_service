import config
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci
import pickle
import os
from QA_dnn.recall.BM25vectorizer import BM25Vectorizer
from QA_dnn.recall.FastText_vectorizer import FastTextVectorizer


class Sentence2Vector(object):
    def __init__(self, method="FastText", byword=False):
        self.byword = byword
        # TODO byword：没有使用byword进行建立索引，所有索引都是按照词语建立的
        self.method = method
        self.qa_dict = json.load(open(config.chatbot_QA_dict_json_path, encoding="utf-8"))
        # self.qa_dict = json.loads(open(config.chatbot_QA_dict_text_path, encoding="utf-8").readline())
        self.index_path = config.recall_search_index_path
        if method == "BM25":
            self.vectorizer = BM25Vectorizer()
            self.index_path = self.index_path + "_bm25"
        elif method == "FastText":
            self.vectorizer = FastTextVectorizer()
            self.index_path = self.index_path + "_fasttext"
        elif method == "Tfidf":
            self.vectorizer = TfidfVectorizer()
            self.index_path = self.index_path + "_tfidf"

    def build_vector(self):

        if self.method in ["FastText", "BM25", "Tfidf"]:
            lines = [q for q in self.qa_dict]
            lines_cuted = [" ".join(self.qa_dict[q]["q_cut"]) for q in lines]  # [sentence1, sentence2,.....]
            features_vec = self.vectorizer.fit_transform(lines_cuted)
            # vectorizer，后续还需要对用户输入的问题进行同样的处理
            search_index = self.get_search_index(features_vec, lines)
        else:
            raise KeyError("method error")
        return self.vectorizer, features_vec, lines_cuted, search_index

    def build_search_index(self, vector, data):
        search_index = ci.MultiClusterIndex(vector, data)
        pickle.dump(search_index, open(self.index_path, "wb"))
        return search_index

    def get_search_index(self, vector, data):
        if os.path.exists(self.index_path):
            search_index = pickle.load(open(self.index_path, mode="rb"))
        else:
            search_index = self.build_search_index(vector, data)

        return search_index
