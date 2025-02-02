# 타이타닉 생존자 예측하기

## 데이터 로드하기 


```python
import pandas as pd
import numpy as np

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
```


```python
# 아래는 데이터의 자세한 정보를 출력하는 코드이다. 괄호 안에 숫자를 삽입하지 않을 시 기본적으로 5줄을 출력한다.
train.head(5)
```

## 데이터 분석하기 


```python
print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
print('----------[train infomation]----------')
print(train.info())
print('----------[test infomation]----------')
print(test.info())
```


```python
# train 데이터 내 카테코리별 NaN 값의 합
train.isnull().sum()
```


```python
# test 데이터 내의 카테고리별 NaN 값의 합
test.isnull().sum()
```


```python
# 시각화 데이터 출력을 위한 코드
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
```

## 데이터 차트로 훑어보기


```python
def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    
    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()
    
    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')
    
    plt.show()
```


```python
pie_chart('Sex')
```


```python
pie_chart('Pclass')
```


```python
pie_chart('Embarked')
```


```python
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
```


```python
bar_chart("SibSp")
```


```python
bar_chart("Parch")
```


```python
train.head(5)
```

## 데이터 전처리와 특성 추출하기


```python
# 두 개의 데이터 병합하기
train_and_test = [train, test]

for dataset in train_and_test:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

train.head(5)
```


```python
pd.crosstab(train['Title'], train['Sex'])
```


```python
# 중복되는 데이터 통일하기
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer',
                                                 'Lady','Major', 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```


```python
# String 데이터로 변환하기
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str)

train.head(5)
```


```python
for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)
    
train.head(26)
```


```python
train.Embarked.unique()
```


```python
train.Embarked.value_counts(dropna=False)
```


```python
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
```


```python
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].astype(str)
```


```python
for dataset in train_and_test:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)
print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
```


```python
#pd.options.mode.chained_assignment = None
#title_list = train.Title.unique()
#for dataset in train_and_test:
#    for title in title_list:
#        age_mean = dataset[dataset['Title'] == title]['Age'].mean()
#        dataset[dataset['Title'] == title]['Age'].fillna(age_mean, inplace=True)
#        print(dataset[dataset['Title'] == title]['Age'])
#        
#    print(dataset['Age'].isnull().sum())
#    #dataset['Age'] = dataset['Age'].astype(int)
#
#train['AgeBand'] = pd.cut(train['Age'], 5)
#print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
```


```python
for dataset in train_and_test:
    dataset.loc[ dataset['Age'] > 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)
```


```python
# Pclass와 Fare의 연관성을 바탕으로 Fare 데이터가 누락된 Pclass를 가진 사람들에게 평균 Fare를 넣어주기
print (train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test["Fare"].isnull()]["Pclass"])
```


```python
for dataset in train_and_test:
    dataset['Fare'] = dataset['Fare'].fillna(13.675)
```


```python
for dataset in train_and_test:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train.head()
```


```python
for dataset in train_and_test:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
    dataset['Family'] = dataset['Family'].astype(int)
```


```python
# 평가에서 제외할 카테고리 추출하기
features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'Fare'], axis=1)

print(train.head())
print(test.head())
```


```python

```


```python
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop("PassengerId", axis=1).copy()

print(train_data.shape, train_label.shape, test_data.shape)
```

## 모델 설계 및 학습하기


```python
# 사이킷-런 라이브러리 로드하기
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle
```


```python
train_data, train_label = shuffle(train_data, train_label, random_state = 5)
```


```python
# 모델 학습 평가에 필요한 파이프라인 제작
def train_and_test(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction
```


```python
# Logistic Regression
log_pred = train_and_test(LogisticRegression())
# SVM
svm_pred = train_and_test(SVC())
#kNN
knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))
# Random Forest
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
# Navie Bayes
nb_pred = train_and_test(GaussianNB())
```


```python
test.head()
```

## CSV 파일 만들기


```python
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": rf_pred
})

submission.to_csv('submission_HBLEE.csv', index=False)
```
