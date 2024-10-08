import os
import numpy as np
from gensim.models import Word2Vec
import preprocess

# 加载预训练的 Word2Vec 模型
model_path = "./Model/mixed_word2vec.model"
word2vec_model = Word2Vec.load(model_path)

# 定义数据目录
safe_dir = "./Data/RowData/Safe"
unsafe_dir = "./Data/RowData/Unsafe"

# 定义函数：将词汇列表转换为文档向量（使用词向量的平均值）
#[``,``...]
def get_document_vector(tokens, model):
    # 获取模型词汇表
    vocab = set(model.wv.key_to_index.keys())
    # 过滤掉不在词汇表中的词
    tokens = [token for token in tokens if token in vocab]
    if len(tokens) == 0:
        # 如果没有有效的词，返回全零向量
        return np.zeros(model.vector_size)
    # 获取每个词的向量
    vectors = [model.wv[token] for token in tokens]
    # 计算平均向量
    document_vector = np.mean(vectors, axis=0)
    return document_vector

# 存储所有的数据
data = []
labels = []

# 处理 Safe 目录下的文件
for i in range(1, 301):
    file_path = os.path.join(safe_dir, f"{i}.txt")
    if os.path.exists(file_path):
        # 预处理文本，得到词汇列表
        tokens = preprocess.preprocess_text(file_path)
        # 将词汇列表转换为文档向量
        doc_vector = get_document_vector(tokens, word2vec_model)
        data.append(doc_vector)
        labels.append(1)
    else:
        print(f"文件 {file_path} 不存在。")

# 处理 Unsafe 目录下的文件
for i in range(1, 301):
    file_path = os.path.join(unsafe_dir, f"{i}.txt")
    if os.path.exists(file_path):
        tokens = preprocess.preprocess_text(file_path)
        doc_vector = get_document_vector(tokens, word2vec_model)
        data.append(doc_vector)
        labels.append(-1)
    else:
        print(f"文件 {file_path} 不存在。")

# 将数据和标签合并，转换为 numpy 数组
data = np.array(data)
labels = np.array(labels)

# 保存数据到文件
output_data_path = "./Data/data_vectors.npy"
output_labels_path = "./Data/data_labels.npy"

np.save(output_data_path, data)
np.save(output_labels_path, labels)

print(f"数据已保存到 {output_data_path} 和 {output_labels_path}")
