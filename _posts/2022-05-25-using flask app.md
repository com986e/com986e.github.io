# **머신러닝 모델을 사용하여 웹앱 만들기**

**학습할 내용:**
* 훈련된 모델을 Pickle(집어내기)하는 방법
* Flask 앱에서 해당 모델을 사용하는 방법

**사용되는 도구**
* Flask : 제작자가 'micro-framework'로 정의한 Flask는 Python을 사용하여 웹 프레임워크의 기본 기능과 웹 페이지를 생성하는 템플릿 엔진을 제공
* Pickle : 모델을 'pickle'하면 웹에서 사용하기 위해 구조를 직렬화하거나 평면화함

**데이터 정리하기**

이 활동에서는 80,000회 이상의 UFO 목격 데이터를 사용한다.


```python
import pandas as pd
import numpy as np

ufos = pd.read_csv('../input/hknu-ml-exercise9/ufos.csv')
ufos.head()
```

![KakaoTalk_20220526_011327821](https://user-images.githubusercontent.com/79626396/170310816-9affbf4e-a7a5-462e-93cc-04cec6e58426.png)
UFO 목격 데이터를 작은 데이터프레임으로 변환한 후 Country 카테고리의 고유값을 확인한다.

```python
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

ufos.Country.unique()
```

![KakaoTalk_20220526_011346749](https://user-images.githubusercontent.com/79626396/170310851-c38e65d8-e02d-40ea-a1ec-aa41d2c348e7.png)

NaN 값을 삭제한 후, 1초 ~ 60초 내의 목격 데이터만 로드하여 처리할 하는 데이터의 양을 경감한다.


```python
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

![KakaoTalk_20220526_011409075](https://user-images.githubusercontent.com/79626396/170310871-57b9bbae-90f2-45f6-98df-b47ac127b161.png)

LabelEncoder : 국가의 텍스트 값을 숫자로 치환하기 위하여 사이킷-런의 라이브러리를 로드한다.

참고 : LabelEncoder는 데이터를 알파벳 순서로 인코딩한다.


```python
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```

![KakaoTalk_20220526_011431345](https://user-images.githubusercontent.com/79626396/170310887-be5421c1-dc49-4935-af1c-37c3fc19b87c.png)

**모델 구축하기**

데이터를 훈련 세트와 테스트 세트로 나누어 모델 훈련 준비를 진행한다.


```python
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

logistic regression을 사용하여 모델을 훈련한다.


```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

![KakaoTalk_20220526_011459996](https://user-images.githubusercontent.com/79626396/170310922-930ce27a-a259-4db5-821d-d0c9cdb47e34.png)

**모델을 Pickle하기**

모델에 Pickle(집어내기)을 진행해보자.

pickle 되면, pickle된 모델을 불러와서 초, 위도와 경도 값이 포함된 샘플 데이터 배열을 대상으로 테스트한다.


```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

![KakaoTalk_20220526_011526383](https://user-images.githubusercontent.com/79626396/170310938-b961cc2f-de41-4598-acfd-1e88762b787f.png)

**Flask 앱 만들기**

1. ufo-model.pkl 파일과 notebook.ipynb 파일 옆에 web-app 이라는 이름의 폴더를 만든다.

2. 3가지 폴더를 만든다: static, 내부에 css 파일이 있는 폴더, templates 폴더. 지금부터 다음 파일과 디렉토리들이 있어야 합니다:


```python
# web-app/
#   static/
#     css/
#     templates/
# notebook.ipynb
# ufo-model.pkl
```

3. web-app 폴더에서 만들 파일은 requirements.txt 파일이다.


```python
# scikit-learn
# pandas
# numpy
# flask
```

4. 지금부터, web-app 으로 이동해서 파일을 실행한다:


```python
# cd web-app
```

5. 터미널에서 pip install 을 입력하고, requirements.txt 에 라이브러리를 설치한다:


```python
# pip install -r requirements.txt
```

6. 지금부터, 앱을 완성하기 위해서 3가지 파일을 더 만든다:


```python
# a. 최상단에 app.py를 만든다.
# b. templates 디렉토리에 index.html을 만든다.
# c. static/css 디렉토리에 styles.css를 만든다.
```

7. 일정한 서식의 styles.css 파일을 만든다:

8. 다음 index.html 파일을 만든다:

9. app.py 에 추가합니다:


```python
# import numpy as np
# from flask import Flask, request, render_template
# import pickle

# app = Flask(__name__)

# model = pickle.load(open("./ufo-model.pkl", "rb"))


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


# if __name__ == "__main__":
#     app.run(debug=True)
```

실행 결과는 이러하다.

![KakaoTalk_20220526_011616086](https://user-images.githubusercontent.com/79626396/170310962-86092d46-c45b-41be-9616-9ad7e3499460.png)

