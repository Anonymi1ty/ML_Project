import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# 4. 初始化结果字典
results = {}

# 5. 训练模型并计算指标
for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    
    # 有些模型（如Naive Bayes）不支持 predict_proba，需做兼容处理
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

# 6. 输出结果
print("模型性能比较：\n")
for name, metrics in results.items():
    print(f"模型：{name}")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")
    print()
