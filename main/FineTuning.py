from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import preprocess

# Step 1: 加载预训练的 Google Word2Vec 模型
pretrained_model_path = './Model/GoogleNews-vectors-negative300.bin.gz'
pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)

# Step 2: 定义自己的语料库
new_sentences = []
# 使用 preprocess_text 函数对文本进行预处理
for i in range(1, 301):
    safe_file_path = f"./Data/RowData/Safe/{i}.txt"
    unsafe_file_path = f"./Data/RowData/Unsafe/{i}.txt"
    safe_data = preprocess.preprocess_text(safe_file_path)
    unsafe_data = preprocess.preprocess_text(unsafe_file_path)
    new_sentences.append(safe_data)
    new_sentences.append(unsafe_data)

# Step 3: 初始化新的 Word2Vec 模型，并使用新的语料库构建词汇表
vector_size = pretrained_model.vector_size
new_model = Word2Vec(vector_size=vector_size, min_count=1, window=5, sg=1)
new_model.build_vocab(new_sentences)

# Step 4: 将预训练的词向量加载到新模型中，并锁定这些词的更新
words_in_pretrained = set(pretrained_model.key_to_index.keys())
new_vocab = set(new_model.wv.key_to_index.keys())

# 初始化所有词的 lockf 为1.0，表示默认情况下词向量会被更新
new_model.wv.vectors_lockf = np.ones(len(new_model.wv), dtype=np.float32)

for word in new_vocab:
    index = new_model.wv.key_to_index[word]
    if word in words_in_pretrained:
        # 将预训练的词向量赋值给新模型
        new_model.wv.vectors[index] = pretrained_model[word]
        # 锁定预训练词的更新
        new_model.wv.vectors_lockf[index] = 0.0
    else:
        # 对于新词，保持 lockf 为1.0，使其在训练过程中更新
        pass  # 新词的向量已在 build_vocab 时随机初始化

# Step 5: 继续训练新模型，只更新新词的向量
new_model.train(new_sentences, total_examples=new_model.corpus_count, epochs=10)

# Step 6: 保存最终模型
new_model.save("./Model/mixed_word2vec.model")
