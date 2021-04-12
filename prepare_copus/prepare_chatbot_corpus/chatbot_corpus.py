"""
准备闲聊语料
"""

import config
import string
from lib import cut
from tqdm import tqdm

def filter(pair):
    """

    :param pair: [q, a]
    :return:
    """

    if pair[0][1].strip() in string.ascii_lowercase:  # 过滤掉只有一个字符的问答句子
        return True
    elif pair[1][1].count("=") > 2: # 防止回答是一些 =。=的符号表情
        return True
    # TODO 过滤掉一些特殊字符，如 = ， 。 》等标点符号组成的句子,或者两个以上字母组成的句子，或者其他不需要的句子
    # elif "黄鸡" in pair[0][1] or "黄鸡" in pair[1][1]:  # 过滤带“黄鸡”的问答对
    #     return True

def prepare_xiaohuangji(by_word=False, use_stopwords=False):
    path = config.xiaohuangji_path
    if by_word:
        input_path = config.chat_input_byword_path
        target_path = config.chat_target_byword_path
    else:
        input_path = config.chat_input_path
        target_path = config.chat_target_path

    f_input = open(input_path, "a", encoding="utf-8")
    f_target = open(target_path, "a", encoding="utf-8")

    one_qa_pair = []  # 保存一个问答对
    num = 0
    for line in tqdm(open(path, encoding="utf-8").readlines(), desc="小黄鸡"):

        if line.startswith("E"):  # 不要E开头的句子
            continue

        else:
            line = line[1:].strip().lower()
            line_cuted = cut(line, by_word, use_stopwords=use_stopwords)
            line_cuted = " ".join(line_cuted)
            if line_cuted.strip() == "":  # 使用停用词后，空行不写入文件,即文件不保留空行
                # print("line:",a)
                # if num == 3:
                #     break
                continue
            if len(one_qa_pair) < 2:  # 保存一个问答对
                one_qa_pair.append([line_cuted, line])
            if len(one_qa_pair) == 2:  # 问答对写入
                # assert len(one_qa_pair) != 2, "error"
                # 判断句子是否需要
                if filter(one_qa_pair):
                    one_qa_pair = []
                    continue

                f_input.write(one_qa_pair[0][0] + "\n")  # 写入提问
                f_target.write(one_qa_pair[1][1] + "\n")  # 写入回答
                # print(one_qa_pair)
                one_qa_pair = []
                num += 1
                # if num == 1000:
                #     break

    f_target.close()
    f_input.close()
    return num