import pandas
from lib import cut
import config
from tqdm import tqdm
import json
import random

# 闲聊语料
xiaohuangji_path = r'G:\OneDrive\progaram\python\test\pytorch\chat_service\corpus\classify_corpus\小黄鸡未分词.conv'

# 问答语料
byhande_path = r"G:\OneDrive\progaram\python\test\pytorch\chat_service\corpus\classify_corpus\手动构造的问题.json"
crawled_path = r"G:\OneDrive\progaram\python\test\pytorch\chat_service\corpus\classify_corpus\爬虫抓取的问题.csv"
flags = [0, 0, 0, 0, 1]  # 1/5作为测试集，4/5作为训练集


def keywords_in_line(sentence):
    keywords_list = ["传智播客", "传智", "黑马程序员", "黑马", "python"
                    "人工智能", "c语言", "c++", "java", "javaee", "前端", "移动开发", "ui",
                     "ue", "大数据", "软件测试", "php", "h5", "产品经理", "linux", "运维", "go语言",
                     "区块链", "影视制作", "pmp", "项目管理", "新媒体", "小程序", "前端"]
    for i in keywords_list:
        if i in sentence:
            return True
        else:
            return False

def process_xiaohuangji(f_train, f_test):
    """
    处理小黄鸡语料
    分词之前，一定要对字符串进行 .strip()操作，去除头尾空白字符
    """
    num_train = 0
    num_test = 0
    for senctence in tqdm(open(xiaohuangji_path, encoding='utf-8').readlines(), desc="小黄鸡"):
        # print(senctence)

        # 只保留M开头的句子
        if senctence.startswith("E"):
            flag = 0
            continue

        if senctence.startswith("M"):  # 出现第一个M
            if flag == 0:
                senctence = senctence[1:].strip()
                flag = 1
            else:  # 第二个M出现
                senctence = senctence[1:].strip()
                flag = 0
                # continue

        # 如果句子长度为1，过滤掉
        if len(senctence) > 1:
            res = cut(senctence.strip(), use_stopwords=config.use_stopwords, by_word=config.by_word)

            if not keywords_in_line(res):  # 去掉含有关键词的句子
                res = " ".join(res) + "\t" + "__label__chat"  # 构造符合FastTest格式的数据
                if random.choice(flags) == 0:
                    f_train.write(res + "\n")
                    num_train += 1
                if random.choice(flags) == 1:
                    f_test.write(res + "\n")
                    num_test += 1
    return num_train, num_test


def process_byhand(f_train, f_test):
    """处理手动构造的数据"""
    num_train = 0
    num_test = 0
    total_lines = json.loads(open(byhande_path, encoding='utf-8').read())
    for key in tqdm(total_lines, desc="byhand"):
        for lines in total_lines[key]:
            for line in lines:
                if "校区" in line:
                    continue
                res = cut(line.strip(), use_stopwords=config.use_stopwords, by_word=config.by_word)
                res = " ".join(res) + "\t" + "__label__QA"  # 构造符合FastTest格式的数据
                if random.choice(flags) == 0:
                    f_train.write(res + "\n")
                    num_train += 1
                if random.choice(flags) == 1:
                    f_test.write(res + "\n")
                    num_test += 1
    return num_train, num_test


def process_crawled_data(f_train, f_test):
    """处理抓取数据"""
    num_train = 0
    num_test = 0
    for line in tqdm(open(crawled_path, encoding='utf-8').readlines(), desc="crawled_data"):
        res = cut(line.strip(), use_stopwords=config.use_stopwords, by_word=config.by_word)
        res = " ".join(res) + "\t" + "__label__QA"  # 构造符合FastTest格式的数据
        if random.choice(flags) == 0:
            f_train.write(res + "\n")
            num_train += 1
        if random.choice(flags) == 1:
            f_test.write(res + "\n")
            num_test += 1
    return num_train, num_test


def process(by_word=False):
    if not by_word:
        f_train = open(config.classify_corpus_train_path, "a", encoding='utf-8')
        f_test = open(config.classify_corpus_test_path, "a", encoding='utf-8')
        # f_train_zhua = open(config.classify_corpus_train_zhua_path, "w", encoding='utf-8')
        # f_test_zhua = open(config.classify_corpus_test_zhua_path, "w", encoding='utf-8')

        # 1.处理小黄鸡
        num_chat_train, num_chat_test = process_xiaohuangji(f_train, f_test)

        # 2.处理手动构造的句子
        num_qa_train_byhand, num_qa_test_byhand = process_byhand(f_train, f_test)

        # # 3.处理抓取的句子
        num_qa_train_crawled, num_qa_test_crawled = process_crawled_data(f_train, f_test)

        f_test.close()
        f_train.close()

        num_qa_test = num_qa_test_byhand + num_qa_test_crawled
        num_qa_train = num_qa_train_byhand + num_qa_train_crawled
        print(num_chat_train, num_chat_test, num_qa_train, num_qa_test)
        print("训练集：", num_chat_train + num_qa_train)
        print("测试集：", num_chat_test + num_qa_test)

    else:
        f_train = open(config.classify_corpus_byword_train_path, "a", encoding='utf-8')
        f_test = open(config.classify_corpus_byword_test_path, "a", encoding='utf-8')
        # f_train_zhua = open(config.classify_corpus_train_zhua_path, "w", encoding='utf-8')
        # f_test_zhua = open(config.classify_corpus_test_zhua_path, "w", encoding='utf-8')

        # 1.处理小黄鸡
        num_chat_train, num_chat_test = process_xiaohuangji(f_train, f_test)

        # 2.处理手动构造的句子
        num_qa_train_byhand, num_qa_test_byhand = process_byhand(f_train, f_test)

        # # 3.处理抓取的句子
        num_qa_train_crawled, num_qa_test_crawled = process_crawled_data(f_train, f_test)

        f_test.close()
        f_train.close()

        num_qa_test = num_qa_test_byhand + num_qa_test_crawled
        num_qa_train = num_qa_train_byhand + num_qa_train_crawled
        print("byword:", num_chat_train, num_chat_test, num_qa_train, num_qa_test)
        print("byword训练集：", num_chat_train + num_qa_train)
        print("byword测试集：", num_chat_test + num_qa_test)

    # f_test_zhua.close()
    # f_train_zhua.close()
# if __name__ == '__main__':
    # process(by_word=True)
    # process(by_word=False)