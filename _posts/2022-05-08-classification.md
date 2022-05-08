# **분류**
---

분류는  지도 학습의 한 형태이며 일반적으로 분류는 이진 분류와 다중 클래스 분류의 두 그룹으로 나뉜다.


*   **선형 회귀** 를 사용하면 변수 사이 관계를 예측하고 새로운 데이터 포인트로 라인과 엮인 위치에 대한 정확한 예측이 가능하다.      ex) 9월과 
12월 각각의 호박 가격에 대하여 예측이 가능하다.
*   **로지스틱 회귀** 는 "이진 범주"를 찾는 데 유용함             ex) 이 가격대에서 이 호박은 주황색인지 초록색인지 구분할 수 있다.

분류는 데이터 포인트의 레이블 또는 클래스를 결정하는 다른 방법을 결정하기 위해 다양한 알고리즘을 사용한다. 이 요리 데이터를 사용하여 재료 그룹을 관찰하여 원산지 요리를 결정할 수 있는지 알아보자.


## 기초작업

### **데이터 불러오기**
---


```python
pip install imblearn
```


```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE
```


```python
df  = pd.read_csv("../input/asian-and-indian-cuisines/asian_indian_recipes.csv")
```


```python
df.head()
```


```python
df.info()
```

### **연습 - 요리에 대해 배우기**
---


```python
df.cuisine.value_counts().plot.barh()
```

요리의 수는 한정되어 있고, 데이터의 분포는 불균형하다.


```python
# 요리별로 사용할 수 있는 데이터 크기
thai_df = df[(df.cuisine == "thai")]
japanese_df = df[(df.cuisine == "japanese")]
chinese_df = df[(df.cuisine == "chinese")]
indian_df = df[(df.cuisine == "indian")]
korean_df = df[(df.cuisine == "korean")]

print(f'thai df: {thai_df.shape}')
print(f'japanese df: {japanese_df.shape}')
print(f'chinese df: {chinese_df.shape}')
print(f'indian df: {indian_df.shape}')
print(f'korean df: {korean_df.shape}')
```

### **성분 찾기**
---

지금부터 데이터를 깊게 분석하여 요리별 일반적인 재료가 무엇인지 알기 위해 요리 사이의 혼동을 일으킬 만한 중복 데이터를 정리해보자.

* Python에서 성분 데이터프레임을 생성하기 위해서 create_ingredient() 함수를 만든다. 이 함수는 도움이 안되는 열을 삭하고 개수별로 재료를 정렬한다.


```python
def create_ingredient_df(df):
    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
    inplace=False)
    return ingredient_df
```

함수를 사용하여 요리별 가장 인기있는 10개 재료에 대한 정보를 얻을 수 있다.


```python
#타이
thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()
```


```python
#일본
japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()
```


```python
#중국
chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()
```


```python
#인도
indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()
```


```python
#한국
korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()
```

 전통 요리 사이에 혼란을 주는 가장 공통적인 재료를 삭제 : 


```python
feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
labels_df = df.cuisine #.unique()
feature_df.head()
```

### **데이터셋 균형 맞추기**
---

데이터를 정리 했으므로 SMOTE ("Synthetic Minority Over-sampling Technique")를 사용하여 균형을 맞춘다.


```python
#fit_resample(): 보간으로 새로운 샘플을 생성함 
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
```

데이터의 균형을 맞추면 분류할 때 더 나은 결과를 얻을 수 있다. 데이터 균형을 맞추면 왜곡된 데이터를 가져와 이러한 불균형을 제거하는 데 도움이 된다.


```python
print(f'new label count: {transformed_label_df.value_counts()}')
print(f'old label count: {df.cuisine.value_counts()}')
```

고르게 균형이 잘 잡혔다.


```python
# 레이블과 특성을 포함한 균형 잡힌 데이터를 파일로 내보낼 수 있는 새 데이터 프레임에 저장
transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
transformed_df
```

기본적으로 Scikit-learn에 로지스틱 회귀를 수행하도록 요청할 때 지정해야 하는 `multi_class` 와 `solver` 중요한 두 개의 파라미터가 있다. 
* `multi_class` 값은 특정 동작을 적용
* `solver`의 값은 사용할 알고리즘


