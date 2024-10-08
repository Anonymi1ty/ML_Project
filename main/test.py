import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# 1. 加载数据
data_vectors = np.load('./Data/data_vectors.npy')
data_labels = np.load('./Data/data_labels.npy')

# 2. 划分训练集和测试集（7:3），并使用 stratify 来保持类别比例
X_train, X_test, y_train, y_test = train_test_split(
    data_vectors, data_labels, test_size=0.3, random_state=42, stratify=data_labels)

# 3. 初始化 XGBoost 模型，监控 error（错误率）
model = XGBClassifier(eval_metric='error', use_label_encoder=False)

# 4. 使用 eval_set 参数来监控训练集和测试集的准确度
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# 5. 获取训练和测试过程中的日志信息
results = model.evals_result()

# 6. 绘制训练集和测试集的准确度变化曲线
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

# 绘制训练集和测试集的 accuracy（准确度）变化曲线
plt.figure(figsize=(10, 6))
plt.plot(x_axis, np.array(results['validation_0']['error']), label='Train error')
plt.plot(x_axis, np.array(results['validation_1']['error']), label='Test error')
plt.title('XGBoost error Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('error')
plt.legend()
plt.grid(True)
plt.show()
