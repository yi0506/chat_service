import fasttext
import config
import numpy as np


def classify_model(by_word=False):
    """
    模型训练
    :param by_word:是否使用单个字作为特征，默认按照词语进行切分
    :return:
    """
    if not by_word:  # 按照词语进行切分
        model = fasttext.train_supervised(config.classify_corpus_train_path, epoch=20,
                                          wordNgrams=config.classify_model_wordNgrams,
                                          minCount=config.classify_model_minCount)
        model.save_model(config.classify_model_path)
    else:  # 单个字切分作为特征
        model = fasttext.train_supervised(config.classify_corpus_byword_train_path, epoch=20,
                                          wordNgrams=config.classify_model_wordNgrams,
                                          minCount=config.classify_model_minCount)
        model.save_model(config.classify_model_byword_path)


def load_classify_model(by_word=False):
    """模型加载"""
    save_path = config.classify_model_path if not by_word else config.classify_model_byword_path
    model = fasttext.load_model(save_path)
    return model


def model_eval(by_word=False):
    """模型评估，获取模型准确率"""
    model = load_classify_model(by_word)
    eval_data_path = config.classify_corpus_byword_test_path if by_word else config.classify_corpus_test_path
    test_data = open(eval_data_path, encoding='utf-8').readlines()  # 特征值与目标值
    sentence = []  # 保存特征值
    target = []  # 保存目标值

    # # 去掉目标值，只保留特征值
    # for i in test_data:
    #     idx = i.strip().rfind("\t")
    #     i = i[:idx]
    #     # print(i)
    #     sentence.append(i)
    # # print(sentence)
    # # 得到预测值
    # pred = model.predict(sentence)
    # y = np.array(pred[0][:])
    # # print(y.shape)
    # y = y.reshape((y.shape[0] * y.shape[1],))
    # # print(y.shape)
    # # print(y)
    #
    # # 取出target值
    #
    # for i in test_data:
    #     idx = i.strip().rfind("\t")
    #     i = i[idx+1:].strip()
    #     target.append(i)
    # # print(np.array(target))
    # target = np.array(target)
    #
    # # 计算准确率
    # # print((y == target).astype(np.float32))
    # acc = (y == target).astype(np.float32).mean()
    # print("acc:", acc)

    for line in test_data:
        line = line.strip()
        # print(line)
        test_data_list = line.split('__label__')
        # print(test_data_list)
        target.append(test_data_list[1].strip())  # 取出目标值
        sentence.append(test_data_list[0].strip())  # 取出特征值

    # 对特征进行模型预测
    label, acc_list = model.predict(sentence)

    # 计算准确率
    num = 0
    # print(len(label), len(target))
    for i, j in zip(label, target):
        # print(i, j)
        if i[0].replace("__label__", "") == j:
            num += 1

    acc = num/len(label)  # 平均的准确率
    print(acc)






