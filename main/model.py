import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

# 1. 加载数据
data_vectors = np.load('./Data/data_vectors.npy')
data_labels = np.load('./Data/data_labels.npy')


# 2. 划分训练集和测试集（7:3），并使用 stratify 来保持类别比例
X_train, X_test, y_train, y_test = train_test_split(
    data_vectors, data_labels, test_size=0.3, random_state=42, stratify=data_labels)

# 3. 定义模型列表
models = {
    'SVM': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# 4. 初始化结果字典
results = {}

# 5. 训练模型并计算指标
plt.figure(figsize=(10, 8))  # 初始化一个画布用于绘制 ROC 曲线
# plt.figure(figsize=(12, 10))

for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)

    # SVM 和 Logistic Regression 需要 predict_proba 来计算 ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]  # 获取正类的概率
    else:
        y_proba = np.zeros_like(y_pred)  # 如果没有 predict_proba，赋予默认值（全部为0）

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else "N/A"

    # 保存结果
    results[name] = {
        'Accuracy': accuracy,
        'Recall': recall,
        'F1-score': f1,
        'ROC AUC': roc_auc
    }

    # 计算并绘制 ROC 曲线
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# 6. 绘制 ROC 曲线
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)  # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.savefig('accuracy_over_epochs.png')

# 7. 输出结果
print("model comparison:\n")
for name, metrics in results.items():
    print(f"model:{name}")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")
    print()
