from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from Utils import preprocess

# Step 1: 加载预训练的 Google Word2Vec 模型
# 假设你已经下载了 Google 的预训练模型文件：GoogleNews-vectors-negative300.bin
pretrained_model_path = './Model/GoogleNews-vectors-negative300.bin.gz'
pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)

# Step 2: 定义自己的语料库
new_sentences = []

# Step 3: 初始化一个新的 Word2Vec 模型，并使用新的语料库进行训练
# 在这里你可以设置和预训练模型相同的参数，例如向量维度
vector_size = pretrained_model.vector_size
new_model = Word2Vec(vector_size=vector_size, min_count=1, window=5, sg=1)
new_model.build_vocab(new_sentences)

# Step 4: 将新的词汇和旧的预训练模型进行混合
# 对于预训练模型中已有的词，直接复制词向量
words_in_pretrained = set(pretrained_model.key_to_index.keys())
new_vocab = set(new_model.wv.key_to_index.keys())

# 使用新的模型的词汇表更新预训练模型
for word in new_vocab:
    if word in words_in_pretrained:
        new_model.wv[word] = pretrained_model[word]
    else:
        # 对于新的词，使用初始化的随机向量（或在微调过程中更新它们）
        new_model.wv[word] = np.random.uniform(-0.25, 0.25, vector_size)

# Step 5: 继续训练（微调）新模型，只更新新的词汇
# 通过训练，只对新词汇的向量进行微调
new_model.train(new_sentences, total_examples=new_model.corpus_count, epochs=10)

# Step 6: 保存最终模型
new_model.save("mixed_word2vec.model")
