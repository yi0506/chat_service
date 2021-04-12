
import pandas
import json
from lib import cut
from lib.with_sg import sg
import config

def process_corpus():
    json_path1 = r"G:\json语料\webtext2019zh\web_text_zh_testa.json"
    json_path2 = r"G:\json语料\webtext2019zh\web_text_zh_valid.json"
    question_text = open(config.chatbot_QA_question_text_path, encoding="utf-8", mode="a")
    answer_text = open(config.chatbot_QA_answer_text_path, mode="a", encoding="utf-8")

    with open(json_path1, encoding="utf-8") as f1:
        # 用eval()转化为字典格式
        while True:
            # print(f1.readline())
            data = f1.readline()
            if len(data) == 0:
                break
            _dict = eval(data)
            question = _dict["title"].strip().replace("\n", "").replace("\r", "")
            answer = _dict["content"].strip().replace("\n", "").replace("\r", "")
            answer_text.write(answer + '\n')
            question_text.write(question + "\n")

    # 用json.loads()转化为字典格式
    with open(json_path2, encoding="utf-8") as f2:
        while True:
            data = f2.readline()
            if len(data) == 0:
                break
            _dict = json.loads(data)
            question = _dict["title"].strip().replace("\n", "").replace("\r", "")
            answer = _dict["content"].strip().replace("\n", "").replace("\r", "")
            answer_text.write(answer + "\n")
            question_text.write(question + "\n")
            if len(_dict) == 0:
                break
    question_text.close()
    answer_text.close()


def build_QA_dict_json():
    """
    one_dict
            {
                "问题1":{
                    "主体":["主体1","主体3","主体3"..],
                    "问题1分词后的句子":["word1","word2","word3"...],
                    "答案":"答案"
                        },

                "问题2":{
                    ...
                        }
            }
    :return:
    """
    qa_dict = {}
    question = open(config.chatbot_QA_question_text_path, encoding="utf-8")
    answer = open(config.chatbot_QA_answer_text_path, encoding="utf-8")
    while True:
    # for i in range(10000):
        q_line = question.readline()
        a_line = answer.readline()

        if len(a_line) == 0 or len(q_line) == 0:
            break

        # 构造QA字典
        qa_dict[q_line.strip()] = {}
        qa_dict[q_line.strip()]["answer"] = a_line.strip()
        # print(q_line)
        result = cut(q_line.strip(), with_sg=True)
        # print(result)
        qa_dict[q_line.strip()]["q_cut"] = [i.word for i in result]
        # print([i.word for i in result])
        qa_dict[q_line.strip()]["q_cut_byword"] = cut(q_line.strip(), by_word=True)  # 按照单个字切分

        entity = [i.word for i in result if i.flag in sg]  # 获取为主语的词
        # print(entity)
        qa_dict[q_line.strip()]["entity"] = entity
        # print(qa_dict)
        # break

    question.close()
    answer.close()
    json.dump(qa_dict, open(config.chatbot_QA_dict_json_path, encoding="utf-8", mode="w"), ensure_ascii=False, indent=2)


def build_QA_dict_txt():
    """
        处理找回需要的数据
        {
                "问题1": "问题"
                "主体":["主体1","主体3","主体3"..],
                "问题1分词后的句子":["word1","word2","word3"...],
                "答案1":"答案"
            }
    """

    question = open(config.chatbot_QA_question_text_path, encoding="utf-8")
    answer = open(config.chatbot_QA_answer_text_path, encoding="utf-8")
    qa_txt = open(config.chatbot_QA_dict_text_path, encoding="utf-8", mode="a")
    while True:
        qa_dict = {}
        q_line = question.readline()
        a_line = answer.readline()

        if len(a_line) == 0 or len(q_line) == 0:
            break

        # 构造QA字典
        qa_dict["question"] = q_line.strip()
        qa_dict["answer"] = a_line.strip()
        # print(q_line)
        result = cut(q_line.strip(), with_sg=True)
        # print(result)
        qa_dict["q_cut"] = [i.word for i in result]
        # print([i.word for i in result])
        qa_dict["q_cut_byword"] = cut(q_line.strip(), by_word=True)  # 按照单个字切分

        entity = [i.word for i in result if i.flag in sg]  # 获取为主语的词
        # print(entity)
        qa_dict["entity"] = entity
        # print(qa_dict)
        # break
        qa_txt.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")

    question.close()
    answer.close()



def excel_QA():
    qa_dict = {}
    df = pandas.read_excel("excel_path")
    for q_line, a_line in zip(df["问题"], df["答案"]):
        # 构造QA字典
        qa_dict[q_line.strip()] = {}
        qa_dict[q_line.strip()]["answer"] = a_line.strip()

        result = cut(q_line.strip(), with_sg=True)
        qa_dict[q_line.strip()]["q_cut"] = [i.word for i in result]
        qa_dict[q_line.strip()]["q_cut_byword"] = cut(q_line.strip(), by_word=True)  # 按照单个字切分

        entity = [i.word for i in result if i[1] in sg]  # 获取为主语的词
        qa_dict[q_line.strip()]["entity"] = entity

def cut_qa(byword=True):
    text_path = r"G:\OneDrive\progaram\python\test\pytorch\chat_service\corpus\QA_corpus\question.txt"
    question_cut_text = open(r"G:\OneDrive\progaram\python\test\pytorch\chat_service\corpus\QA_corpus\q_cut.txt", encoding="utf-8", mode="a")
    question_cut_byword_text = open(r"G:\OneDrive\progaram\python\test\pytorch\chat_service\corpus\QA_corpus\q_cut_byword.txt", encoding="utf-8", mode="a")


    with open(text_path, encoding="utf-8") as f1:
        while True:
        # for i in range(10000):
            sentence = f1.readline()
            if len(sentence) == 0:
                break
            if byword:
                data = " ".join(cut(sentence.strip(), by_word=True))
                question_cut_byword_text.write(data + "\n")
            else:
                data = " ".join(cut(sentence.strip()))
                question_cut_text.write(data + "\n")

    question_cut_text.close()

if __name__ == '__main__':
    process_corpus()



