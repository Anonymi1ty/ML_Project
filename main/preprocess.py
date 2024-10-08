import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
# 第一次使用需要下载nltk的数据
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(file_path):
    # 读取文本内容
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 统一转为小写
    text = text.lower()
    
    # 去除标点符号，但保留'@', ',', '$'
    punctuation_to_remove = string.punctuation.replace('@', '').replace(',', '').replace('$', '')
    text = text.translate(str.maketrans('', '', punctuation_to_remove))
    
    # 进行句子分割
    sentences = sent_tokenize(text)
    
    # # 去除停用词和分词
    stop_words = set(stopwords.words('english'))
    # processed_sentences = [``,``,`....`]
    for sentence in sentences:
        words = word_tokenize(sentence)
        processed_sentences = [word for word in words if word not in stop_words]
        # filtered_words = [word for word in words]
        # processed_sentences.append(filtered_words)
    
    return processed_sentences

# 示例调用
# file_path = "./Data/RowData/Unsafe/148.txt"
# processed_data = preprocess_text(file_path)
# print(processed_data)