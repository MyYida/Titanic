#PassengerId：乘客序号；
#Survived：最终是否存活（1表示存活，0表示未存活）；
#Pclass：舱位，1是头等舱，3是最低等；
#Name：乘客姓名；
#Sex：性别；
#Age：年龄；
#SibSp：一同上船的兄弟姐妹或配偶；
#Parch：一同上船的父母或子女；
#Ticket：船票信息；
#Fare：乘客票价，决定了Pclass的等级；
#Cabin：客舱编号，不同的编号对应不同的位置；
#Embarked：上船地点，主要是S（南安普顿）、C（瑟堡）、Q（皇后镇）。
#导入数据包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#导入训练数据
train = pd.read_csv("D:/download/train.csv")
#导入测试数据
test = pd.read_csv("D:/download/test.csv")
#打印训练数据
#print(train)
#打印测试数据
#print(test)

#获取训练集和数据集数目
#print('训练数据集:', train.shape, '测试数据集:', test.shape)

#获取train数据属性
#print(train.describe())

#补充age缺失数据
# train['Age'] = train['Age'].fillna(train['Age'].median())

#泰坦尼克号整体幸存率
# survived_rate = float(train['Survived'].sum())/train['Survived'].count()
# print(survived_rate)

#幸存者与遇难者可视化
#显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# train.info()
# # 查看幸存人数
# survived_num = train['Survived'].sum()
# victim_num = 891 - train['Survived'].sum()
#
# plt.figure(figsize=(10, 5))#指定图片大小
# plt.subplot(1, 2, 1)#表示在本区域里显示1行2列个图像，最后的1表示本图像显示在第一个位置。
# sns.countplot(x="Survived", data=train)
# plt.title("幸存者数量")
#
# plt.subplot(1, 2, 2)
# plt.pie([survived_num, victim_num], labels=['幸存者', '遇难者'], autopct='%1.0f%%')
# plt.title('遇难者数量')
# plt.show()


#数据分析

# # 1.幸存者与舱位（Pclass）的关系
# pclass_1 = train[train['Pclass']==1]
# pclass_2 = train[train['Pclass']==2]
# pclass_3 = train[train['Pclass']==3]
#
# plt.figure(figsize=(10, 20))
# plt.subplot(4, 2, 1)
# sns.countplot(x='Survived', data=pclass_1)
# plt.title('头等舱')
# plt.subplot(4, 2, 2)
# plt.pie([pclass_1['Survived'][pclass_1['Survived']==0].count(), pclass_1['Survived'][pclass_1['Survived']==1].count()], labels=['遇难者', '幸存者'], autopct='%1.0f%%')
#
# plt.figure(figsize=(10, 20))
# plt.subplot(4, 2, 3)
# sns.countplot(x='Survived', data=pclass_2)
# plt.title('中等舱')
# plt.subplot(4, 2, 4)
# plt.pie([pclass_2['Survived'][pclass_2['Survived']==0].count(), pclass_2['Survived'][pclass_2['Survived']==1].count()], labels=['遇难者', '幸存者'], autopct='%1.0f%%')
#
# plt.figure(figsize=(10, 20))
# plt.subplot(4, 2, 5)
# sns.countplot(x='Survived', data=pclass_3)
# plt.title('低等舱')
# plt.subplot(4, 2, 6)
# plt.pie([pclass_3['Survived'][pclass_3['Survived']==0].count(), pclass_3['Survived'][pclass_3['Survived']==1].count()], labels=['遇难者', '幸存者'], autopct='%1.0f%%')
#
# plt.subplot(4, 1, 4)
# train.groupby('Pclass')['Survived'].mean().plot(kind='bar')
#
# plt.show()
#可以看出船舱的等级越高，生还率越高

#2.幸存者与性别（Sex）的关系

#对男性
# male_train = train[train['Sex']=='male']
# male_train['Survived'][male_train['Survived']==1].count()
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# sns.countplot(x='Survived', data=male_train)
# plt.subplot(122)
# plt.pie([male_train['Survived'][male_train['Survived']==0].count(),male_train['Survived'][male_train['Survived']==1].count()], labels=['遇难者', '幸存者'], autopct='%1.0f%%')
# plt.show()

# #对女性
# female_train = train[train['Sex']=='female']
# female_train['Survived'][female_train['Survived']==1].count()
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# sns.countplot(x='Survived', data=female_train)
# plt.subplot(122)
# plt.pie([female_train['Survived'][female_train['Survived']==0].count(), female_train['Survived'][female_train['Survived']==1].count()], labels=['遇难者', '幸存者'], autopct='%1.0f%%')
# plt.show()

#汇总
# train.groupby('Sex')['Survived'].mean()
# train.groupby('Sex')['Survived'].mean().plot(kind='bar')
# plt.show()
# 女性存活率明显高于男性

# 3.幸存者与年龄（Age）的关系
# plt.figure(figsize=(12, 5))#指定图片大小
# plt.subplot(121)#表示在本区域里显示1行2列个图像，最后的1表示本图像显示在第一个位置。
# train['Age'].hist(bins=70)
# plt.xlabel('年龄')
# plt.ylabel('人数')
# plt.subplot(1, 2, 2)
# plt.show()

#幸存者与有兄弟姐妹或配偶（SibSp)的关系
sibsp_train = train[train['SibSp']!=0]
no_sibsp_train = train[train['SibSp']==0]
# 有兄弟姐妹或配偶
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# sns.countplot(x='Survived', data=sibsp_train)
# plt.subplot(122)
# plt.pie([sibsp_train['Survived'][sibsp_train['Survived']==0].count(), sibsp_train['Survived'][sibsp_train['Survived']==1].count()], labels=['遇难者', '幸存者'], autopct='%1.0f%%')
# plt.show()
#没有兄弟姐妹或配偶
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='Survived', data=no_sibsp_train)
plt.subplot(122)
plt.pie([no_sibsp_train['Survived'][no_sibsp_train['Survived']==0].count(), no_sibsp_train['Survived'][no_sibsp_train['Survived']==1].count()], labels=['遇难者', '幸存者'], autopct='%1.0f%%')
plt.show()
#由图可以看出有兄弟姐妹或配偶的人中生还率为47%，没有兄弟姐妹或配偶的人中生还率为36%，所以有兄弟姐妹或配偶的人生还率较高

#幸存者与有无父母（Parch)的关系
parch_train = train[train['Parch']!=0]
no_parch_train = train[train['Parch']==0]
# 有父母的
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# sns.countplot(x='Survived', data=parch_train)
# plt.subplot(122)
# plt.pie([parch_train['Survived'][parch_train['Survived']==0].count(), parch_train['Survived'][parch_train['Survived']==1].count()], labels=['遇难者', '幸存者'], autopct='%1.0f%%')
# plt.show()
#没有父母
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='Survived', data=no_parch_train)
plt.subplot(122)
plt.pie([no_parch_train['Survived'][no_parch_train['Survived']==0].count(), no_parch_train['Survived'][no_parch_train['Survived']==1].count()], labels=['遇难者', '幸存者'], autopct='%1.0f%%')
plt.show()
#由图可以看出有父母的人生还率为51%，没有父母的人中生还率为34%，所以，有父母的人生还率较高



