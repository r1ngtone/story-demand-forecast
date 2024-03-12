import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def getdata(path):# 导入数据
    data = pd.read_csv(path)# 'FMCG_data.csv'

    # 探索数据
    return(data.head(),'/n',data.describe())
    
getdata('FMCG_data.csv')

# # 数据可视化
# sns.pairplot(data)
# plt.show()

# # 数据准备
# X = data[['feature1', 'feature2', ...]]  # 根据你的数据选择特征列
# y = data['target_column']  # 根据你的数据选择目标列

# # 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 建立模型
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 模型评估
# print("训练集得分:", model.score(X_train, y_train))
# print("测试集得分:", model.score(X_test, y_test))

# # 可视化模型预测
# predictions = model.predict(X_test)
# plt.scatter(y_test, predictions)
# plt.xlabel('实际值')
# plt.ylabel('预测值')
# plt.show()
