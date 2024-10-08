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
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

# 1. 加载数据
data_vectors = np.load('./Data/data_vectors.npy')
data_labels = np.load('./Data/data_labels.npy')

# 将 -1 标签转换为 0
data_labels = np.where(data_labels == -1, 0, data_labels)

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

for name, model in models.items():
    # 对于 SVM 和 KNN，使用概率校准
    if name in ['SVM', 'KNN']:
        model = CalibratedClassifierCV(base_estimator=model, cv=5)

    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)

    # 获取预测概率或决策函数
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
        # 将决策函数的输出映射到 [0, 1] 区间
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    else:
        y_proba = None

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = "N/A"

    # 保存结果
    results[name] = {
        'Accuracy': accuracy,
        'Recall': recall,
        'F1-score': f1,
        'ROC AUC': roc_auc
    }

    # 计算并绘制 ROC 曲线
    if y_proba is not None:
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
plt.show()

# 7. 输出结果
print("模型性能比较：\n")
for name, metrics in results.items():
    print(f"模型：{name}")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")
    print()
