"""测试分类模型的API"""

from classify.build_model import classify_model, load_classify_model, model_eval
import config
import tqdm
import numpy as np
from classify.classify import Classify

if __name__ == '__main__':

    classify_model(True)
    classify_model(False)
    model_eval(config.by_word)
    # not by_word 0.999949725753988
    #     by_word 0.9999681640189743

    # classify = Classify()
    # sentence = {
    #     "cut": "python",
    #     "cut_byword": "python"
    # }
    # ret = classify.predict(sentence)
    # print(ret)

