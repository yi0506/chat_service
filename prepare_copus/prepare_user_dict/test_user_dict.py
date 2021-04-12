"""
测试用户词典
"""
import jieba
import config

# jieba加载用户词典
jieba.load_userdict(config.user_dict_path)

def test_user_dict():
    sentence = 'python+人工智能和c++哪个难'
    res = jieba.lcut(sentence)
    print(res)