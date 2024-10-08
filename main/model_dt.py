from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
data = np.load('./Data/data_vectors.npy')
labels = np.load('./Data/data_labels.npy')

# 按照7:3的比例划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# 初始化决策树模型
decision_tree_model = DecisionTreeClassifier(random_state=42)

# 训练模型
decision_tree_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = decision_tree_model.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# 输出结果
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

