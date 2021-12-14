import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model

#使用RandomForestClassifier
def set_missing_ages(df):
    #把已有的数值型特征取出丢进Random Forest Regressor
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    #乘客年龄分为两类，已知和未知
    know_age = age_df[age_df.Age.notnull()].values
    unknow_age = age_df[age_df.Age.isnull()].values

    #y即目标年龄
    y = know_age[:, 0]
    #x即特征属性值
    x = know_age[:, 1:]

    #fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    #用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknow_age[:, 1::])

    #用得到的预测结果填补缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

train = pd.read_csv("D:/download/train.csv")
print(train)

#填补缺失的年龄属性
train, rfr = set_missing_ages((train))
train = set_Cabin_type(train)

dummies_Cabin = pd.get_dummies(train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(train['Pclass'], prefix='Pclass')
df = pd.concat([train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

scaler = preprocessing.StandardScaler()

#用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

#y即Survived结果
y = train_np[:, 0]
#x即特征属性值
x = train_np[:, 1:]
#fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6, solver='liblinear')
clf.fit(x, y)


test = pd.read_csv("D:/download/test.csv")
test.loc[(test.Fare.isnull()), 'Fare'] = 0
#对test做和train一致的特征变换
#首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[test.Age.isnull()].values
#根据特征属性x预测年龄并补上
x = null_age[:, 1:]
predictedAges = rfr.predict(x)
test.loc[(test.Age.isnull()), 'Age'] = predictedAges

test = set_Cabin_type(test)
dummies_Cabin = pd.get_dummies(test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(test['Pclass'], prefix='Pclass')

df_test = pd.concat([test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

test1 = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test1)
result = pd.DataFrame({'PassengerId': test['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
result.to_csv("D:/download/logistic_regression_predictions.csv", index=False)



