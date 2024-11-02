import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 读取CSV文件
labels_df = pd.read_csv('git_target.csv')
embeddings_df = pd.read_csv('git_embedding.csv')

# 确保两份数据都以'id'为索引，以便于合并
labels_df.set_index('id', inplace=True)
embeddings_df.set_index('id', inplace=True)

# 合并数据集，基于'id'
merged_df = embeddings_df.join(labels_df, how='inner')

# 检查是否有任何行在合并后丢失了标签
if merged_df['ml_target'].isnull().any():
    print("警告：某些节点没有对应的标签，请检查原始数据。")
    # 可以选择删除这些行或者填充缺失值
    merged_df.dropna(subset=['ml_target'], inplace=True)

# 提取特征（embedding）和目标变量
X = merged_df.drop(columns=['name', 'ml_target']).values  # 嵌入向量
y = merged_df['ml_target'].values  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 现在X_train, X_test, y_train, y_test已经准备好用于训练模型
print(f"训练样本数量: {len(X_train)}")
print(f"测试样本数量: {len(X_test)}")

# 输出一些基本信息
print(merged_df.head())

# 创建逻辑回归模型，并设置L2正则化
model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted') # 对于多分类问题

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')