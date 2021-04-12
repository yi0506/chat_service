"""配置文件"""
import pickle
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############# 语料相关 ###################
user_dict_path = "corpus/user_dict/keywords.txt"
stopwords_path = 'corpus/user_dict/stopwords.txt'

classify_corpus_train_path = "corpus/classify_corpus/train.txt"
classify_corpus_test_path = "corpus/classify_corpus/test.txt"
classify_corpus_byword_train_path = "corpus/classify_corpus/train_byword.txt"
classify_corpus_byword_test_path = "corpus/classify_corpus/test_byword.txt"

use_stopwords = False
by_word = True

################## 分类相关 #####################
classify_model_path = "model/classify/classify.model"  # 词语作为特征的模型
classify_model_byword_path = "model/classify/classify_byword.model"  # 单个字作为特征的模型

classify_model_wordNgrams = 2
classify_model_minCount = 1

final_model_path = 'model/classify/classify_byword.model'  # 最终选择的模型,单个字为特征
final_model__byword_path = 'model/classify/classify.model'  # 最终选择的模型,一个词为特征



################## 闲聊chatbot相关#####################

xiaohuangji_path = "corpus/classify_corpus/小黄鸡未分词.conv"
# chat_byword = True
# if chat_byword:
chat_input_byword_path = "corpus/chatbot/input_byword.txt"
chat_target_byword_path = "corpus/chatbot/target_byword.txt"
# else:
chat_input_path = "corpus/chatbot/input.txt"
chat_target_path = "corpus/chatbot/target.txt"

chatbot_batch_size = 1024
chatbot_byword_max_len = 30
chatbot_max_len = 14



#### ws  ######

chat_ws_target_byword_path = "model/chatbot/ws_target_byword.pkl"
chat_ws_input_byword_path = "model/chatbot/ws_input_byword.pkl"
# else:
chat_ws_input_path = "model/chatbot/ws_input.pkl"
chat_ws_target_path = "model/chatbot/ws_target.pkl"


chatbot_ws_input = pickle.load(open(chat_ws_input_byword_path, "rb"))
chatbot_ws_target = pickle.load(open(chat_ws_target_byword_path, "rb"))

############### 闲聊模型 ################
chatbot_embedding_dim = 256
chatbot_encoder_num_layer = 2
chatbot_encoder_hidden_size = 128

beam_width = 3
clip = 0.01

chatbot_decoder_num_layer = 2
chatbot_decoder_hidden_size = 128

chatbot_teacher_forcing_ratio = 0.7

chatbot_seq2seq_model_save_path = "model/chatbot/seq2seq.model"
chatbot_seq2seq_optimizer_save_path = "model/chatbot/seq2seq.optimizer"
chatbot_seq2seq_byword_model_save_path = "model/chatbot/seq2seq_byword.model"
chatbot_seq2seq_byword_optimizer_save_path = "model/chatbot/seq2seq_byword.optimizer"


############# QA_chatbot_recall# #############
chatbot_QA_question_text_path = "corpus/QA_corpus/question.txt"
chatbot_QA_answer_text_path = "corpus/QA_corpus/answer.txt"
chatbot_QA_dict_json_path = "corpus/QA_corpus/qa_dict.json"
chatbot_QA_dict_text_path = "corpus/QA_corpus/qa_dict.txt"
recall_search_index_path = "model/recall_dnn/search_inex"
recall_topk = 5
recall_clusters = 10

fasttext_model_path = "model/recall_dnn/fasttext.model"
fasttext_model_byword_path = "model/recall_dnn/fasttext_byword.model"

################ QA_chatbot_sort #################
sort_ws_save_path = 'model/sort_dnn/sort_ws.pkl'
sort_q_cut_path = "corpus/QA_corpus/q_cut.txt"
sort_q_cut_similar_path = "corpus/QA_corpus/q_cut_similar.txt"
sort_q_cut_target_path = "corpus/QA_corpus/q_cut_target.txt"

sort_ws = pickle.load(open(sort_ws_save_path, "rb"))
sort_batch_size = 128
sort_max_len = 30
sort_similar_max_len = 30
siamese_embedding_dim = 300
siamese_hidden_size = 256
siamese_num_layer = 2
bidirectional = True
pooling_stride = 2
pooling_kernal_size = 2
siamese_drop_out = 0.3

sort_model_path = 'model/sort_dnn/model.ckpt'