from prepare_copus.prepare_user_dict.test_user_dict import test_user_dict
import numpy as np
from lib.cut_sentence import cut
from lib import stopwords
import config
from prepare_copus.prepare_classify_corpus.build_classify import process, process_xiaohuangji
from prepare_copus.prepare_chatbot_corpus.chatbot_corpus import prepare_xiaohuangji
from prepare_copus.prepare_QA_corpus.recall_corpus import build_QA_dict_txt, build_QA_dict_json
from prepare_copus.prepare_QA_corpus.recall_corpus import process_corpus, cut_qa

if __name__ == '__main__':
    # sentence = "'python难不难？是不是很难。只要', '只有', '至', '至于', '诸位'"
    # print(cut(sentence, with_sg=False, use_stopwords=True))
    # # print(stopwords)
    # process(True)
    # process(False)
    # num = prepare_xiaohuangji(by_word=True)
    # num2 = prepare_xiaohuangji(by_word=False)
    # print(num, num2)      # 454583
    # process_corpus()
    # build_QA_dict_txt()
    build_QA_dict_json()
    # cut_qa(False)
    # cut_qa(byword=True)
