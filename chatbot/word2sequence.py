class Word2Sequence(object):
    PAD_TAG = 'PAD'
    PAD = 0
    UNK_TAG = 'UNK'
    UNK = 1
    SOS_TAG = 'SOS'  # 句子开始的符号
    SOS = 2
    EOS_TAG = "EOS"  # 句子结束的符号
    EOS = 3

    def __init__(self):
        self.dict = {
            Word2Sequence.PAD_TAG: Word2Sequence.PAD,
            Word2Sequence.UNK_TAG: Word2Sequence.UNK,
            Word2Sequence.EOS_TAG: Word2Sequence.EOS,
            Word2Sequence.SOS_TAG: Word2Sequence.SOS
        }
        self.count = {}
        self.inverse_dict = {}

    def transform(self, sentence, max_len, add_eos=False):
        """
        将sentence转化为数字序列,
        当add_eos=True时：句子长度为 max_len + 1

                        eg1:
                        sentence: 11
                        max_len: 10
                        结果：句子长度为11


                        eg2:
                        sentence: 8
                        max_len: 10
                        结果：句子长度为11

        当add_eos=False时：句子长度为 max_len

                        eg1：
                        eg2：
                        结果： 10


        """
        if len(sentence) > max_len:  # 裁剪
            sentence = sentence[:max_len]
        sentence_len = len(sentence)  # 提前计算句子长度，实现add_eos后，句子长度统一
        if add_eos:
            sentence = sentence + [Word2Sequence.EOS_TAG]

        if sentence_len < max_len:  # 填充
            sentence = sentence + [Word2Sequence.PAD_TAG] * (max_len - sentence_len)

        result = [self.dict.get(i, Word2Sequence.UNK) for i in sentence]
        return result

    def inverse_transform(self, indices):
        """把序列转化为字符串"""
        # return [self.inverse_dict.get(i, Word2Sequence.UNK_TAG) for i in indices]
        result = []
        for i in indices:
            if i == Word2Sequence.EOS:
                break
            result.append(self.inverse_dict.get(i, Word2Sequence.UNK_TAG))
        return result

    def fit(self, sentence):
        """
        传入一个句子，统计词频
        :param sentence: [word1, word2,....]
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=5, max_conut=None, max_feature=None):
        """
        构造词典
        :param min_count: 最小的词频
        :param max_conut: 最大的词频数
        :param max_feature: 词典最多有多少个词
        :return:
        """
        # if min_count is not None:
        #     # 删除小于词频min_count的词
        #      self.count = {word: count for key, count in self.count.items() if count >= min_count}
        # if max_conut is not None:
        #     # 删除大于词频max_count的词
        #     self.count = {word: count for word, count in self.count.items() if count <= max_conut}

        temp = self.count.copy()
        for key in temp:  # 这样能少一次for循环
            cur_count = self.count.get(key, 0)
            if cur_count < min_count:
                del self.count[key]  # 删除小于词频max_count的词

            if max_conut is not None:
                if cur_count > max_conut:
                    del self.count[key]  # 删除大于词频min_count的词

        if max_feature is not None:  # 限制保留的词语数
            self.count = dict(sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:max_feature])

        for key in self.count:
            self.dict[key] = len(self.dict)

        self.inverse_dict = dict(zip(self.count.values(), self.count.keys()))

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    print(Word2Sequence().inverse_dict)
