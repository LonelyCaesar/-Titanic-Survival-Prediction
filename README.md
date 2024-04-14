# Titanic-Survival-Prediction
# 一、說明
本專題透過Kaggle用鐵達尼號生存預測比賽，使用探索性數據分析認識資料到特徵工程處理欄位的生存率、產生訓練集與測試集等產生模型預測資料。
此專案研究用意，對於人工智慧與大數據分析初學者建立程度外，使用pyton程式碼及相關套件實作，然後結合Kaggle 討論區中的文章，更深入來優化整個分析過程。
# 二、實作(請使用python3.6版做執行)
透過 Kaggle 資料競賽網站，下載鐵達尼號資料集。(Link: https://www.kaggle.com/c/titanic/data)

(點擊 "Download All" 後解壓縮，並透過下方程式碼上傳 gender_submission.csv, test.csv , train.csv 三份檔案)

※ 可一次上傳或分批上傳，上傳成功後，點擊左方 "Files" 欄位，即可看到上傳的檔案。 ※
### 1.	讀取資料合併：
訓練集有 891 筆資料、12 個特徵欄位； 測試集有 418 筆資料、11 個特徵欄位； 其中，訓練集較測試集多了判別乘客罹難或生還的特徵欄位 Survived，0 表示罹難者，1 表示生還者。
### 程式碼：
```python
# 忽略警告訊息
import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns 
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, linear_model

# 載入數據
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
submit = pd.read_csv('gender_submission.csv')

print('train',df_train.shape)
display(df_train.head(5))
print('test',df_test.shape)
display(df_test.head(5))
```
```python
# 合併train及test的資料 
df_data = df_train.append( df_test )
df_data
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/1f59132d-6a20-4f96-886f-a4bf54cef8ee)

合併後訓練測試集總共有1309筆資料、12個特徵欄位，做出一致性的預測分析及模型訓練就會比較快速好理解。
### 2.	生存率：
首先，我們分析生還者與罹難者的比例是否有明顯極大的落差，比如生還者的比例僅有 1%，若資料有極大的落差時，表示存在『數據不平衡』(Imbalanced Data)的問題，則後續需用特別的方法對資料進行抽樣。 接下來，我們分別觀察性別(Sex)、票務艙(Pclass)、登船港口(Embarked)、兄弟姊妹配偶人數(SibSp)、父母子女人數(Parch)與生存率的關係。
### 程式碼：
```python
sns.countplot(df_data['Sex'], hue=df_data['Survived'])
df_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().round(2)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/e6862fe7-130f-4e3f-859e-50899e95afb3)
性別：大部的男性都罹難(僅剩約 19% 存活)，而女性則大部分都倖存(約 74%)。
### 程式碼：
```python
sns.countplot(df_data['Pclass'], hue=df_data['Survived'])
df_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().round(2)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/31062481-01f6-43c3-b051-d35a3ccccb0c)
艙等：從數據中可發現頭等艙(Pclass=1)的乘客生存機率較高， 可能不論是逃生設備或是沈船訊息都最先傳到頭等艙。
### 程式碼：
```python
sns.countplot(df_data['Embarked'], hue=df_data['Survived'])
df_data[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().round(2)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/78e0b53b-8b21-4907-a3c5-4b5a9a4a746e)
登船港口：依據數據顯示出來的結果為生存率以C最高。
### 程式碼：
```python
sns.countplot(df_data['SibSp'], hue=df_data['Survived'])
df_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().round(2)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/604f67bd-d9ee-45ee-94aa-c75c17609f3b)
當船上的兄弟姐妹配偶人數有1人同行時，則生存率較高。
### 程式碼：
```python
sns.countplot(df_data['Parch'], hue=df_data['Survived'])
df_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().round(2)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/b286a635-eb01-40a7-bbf0-27e98aab9779)
當船上的父母子女人數為 1~3 人時，有較高的生存率。
### 程式碼：
```python
# 轉換性別資料：0->女性，1->男性
df_data['Sex_Code'] = df_data['Sex'].map({'female':1, 'male':0})
df_data.head(2)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/ca8d785b-8238-4ec5-b4ae-9c6692507b06)
### 程式碼：
```python
# 由於票價分布非常廣，所以將票價取 log 後再畫圖
fig, ax = plt.subplots(figsize = (10,4))
df_data['Log(Fare)'] = df_data['Fare'].map(lambda x:np.log10(x) if x>0 else 0)
sns.boxplot(x='Log(Fare)', y='Pclass', hue='Survived', data=df_data, orient='h', 
            ax=ax, palette="Set3")
pd.pivot_table(df_data, values=['Fare'], index=['Pclass'], columns=['Survived'], 
               aggfunc='median').round(3)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/1b46b1a1-48fc-4072-b1f6-e3218302bf81)
票價和艙等都是屬於彰顯乘客社會地位的一個特徵，買票價格較高的乘客，他們的生存機率也較高。
### 程式碼：
```python
# 登船港口(Embarked)只有遺漏少數，直接補上出現次數最多的 S
df_data['Embarked'] = df_data['Embarked'].fillna('S')

# 費用(Fare)也只有遺漏一筆，因此就直接補上平均值
df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].mean())
# 年紀(Age)的遺漏值較多，需好好思考如何填補
print('Age 遺漏筆數：', df_data['Age'].isnull().sum())

# 0->遺漏 Age
df_data['Has_Age'] = df_data['Age'].isnull().map(lambda x: 0 if x else 1)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figwidth(12)
sns.countplot(df_data['Pclass'], hue=df_data['Has_Age'], ax=ax1)
sns.countplot(df_data['Sex'], hue=df_data['Has_Age'], ax=ax2)
pd.crosstab(df_data['Has_Age'], df_data['Sex'], margins=True).round(3)

# sns.countplot(df['Pclass'], hue=df['Has_Age'])
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/6e51d8ce-605b-4d4d-9c7d-8aef13649b9f)
左圖：年紀遺漏值大部分在3等艙，如果年紀是個重要特徵，則我們對3等艙的觀察就會失真。保守的作法是觀察1,2艙等中年紀對存活與否的影響。
右圖：顯示了遺漏值對性別的分布，其中314位女性有53位缺失年齡(16.9%)，577位男性有124位缺失年齡(21.5%)，男性遺漏年紀的比例稍微多一點(~4.6%)。
### 程式碼：
# 填入遺漏值的方式打算使用姓名中同稱謂的平均值來填補
# 取出姓名當中的稱謂
```python
df_data['Title'] = df_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
title = df_data['Title'].value_counts().index
rare = title[5:]
df_data['Title'] = df_data['Title'].map(lambda x: 'Rare' if x in rare else x)
df_data['Title'] = df_data['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')
df_data['Title'] = df_data['Title'].replace(['Lady'], 'Mrs')
sns.factorplot(x='Title', y='Age', kind='box', hue='Pclass', data=df_data, size=6, aspect=1.3)
missing_mask = (df_data['Has_Age'] == 0)
pd.crosstab(df_data[missing_mask]['Pclass'], df_data[missing_mask]['Title'])
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/81c7c6ae-59ca-4a12-845f-d33c1d36a873)
### 程式碼：
```python
df_data['Title'] = df_data.Title.replace( ['Don','Rev','Dr','Major','Lady','Sir','Col','Capt','Countess','Jonkheer','Dona'], 'Rare' )
df_data['Title'] = df_data.Title.replace( ['Ms','Mlle'], 'Miss' )
df_data['Title'] = df_data.Title.replace( 'Mme', 'Mrs' )
df_data['Title'].unique()
```
### 執行結果：
array(['Mr', 'Mrs', 'Miss', 'Master', 'Rare'], dtype=object)

### 程式碼：
```python
# 將年齡的空值填入年齡的中位數
age_median = np.nanmedian(df_data['Age'])
print('年齡中位數=', age_median)
df_data.loc[df_data.Age.isnull(), 'Age'] = age_median
```
### 執行結果：
執行結果：年齡中位數= 28.0。執行訓練出來的結果平均年齡中位數為28歲。
### 3.	產生訓練集(Train)與測試集(Test)：
完成前述的資料分析及特徵工程後，我們就快可以把資料餵入模型進行訓練了！在此之前的最後一個注意步驟，就是需確認每個欄位皆為數值型態，且將資料分割回訓練集(train)與測試集(test)。所以，此時我們先檢視特徵工程後的資料，觀察是否還有需要處理的欄位。
### 程式碼：執行前
```python
print( f'Shape of data after feature engineering = {df_data.shape}' )
df_data.head()
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/f391be3c-4858-4317-85c4-080c37868cb0)
### 程式碼：執行後
```python
for col in ['Survived','Pclass','Sex','Age','Cabin','Ticket']:
    df_data[col] = df_data[col].astype('category').cat.codes
df_data.head()
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/7cc1ccc7-32d7-446f-bf74-c3b48b43a6a2)
檢視後發現，因為我們之前有處理過字串欄位，所以資料中還存有 6 欄類別型態的欄位：Survived、Pclass、Sex、Age、Cabin、Ticket，因此，我們也需要將它們轉換成數值型態欄位：執行後Survived、Pclass、Sex、Age、Cabin、Ticket的Dtype發現字元有變動了，接者就可以進行訓練。
### 4.	訓練模型：
最後將上述分析好的模型數字型態改變後，接者就可以將要訓練的模型欄位名作出預算，來評估模型的準確度是否能達到我們要求的水準。
### 程式碼：執行前
```python
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(df_data["Pclass"])
X = pd.DataFrame([encoded_class, df_data["Sex"], df_data["Age"]]).T
y = df_data["Survived"]
logistic = linear_model.LogisticRegression()
logistic.fit(X, y)
print('截距=',logistic.intercept_)
print('迴歸係數=',logistic.coef_)
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/e3800783-b08a-46cb-92b4-8569d794490a)
### 程式碼：
```python
print('Confusion Matrix')
preds = logistic.predict(X)
print(pd.crosstab(preds, df_data["Survived"]))
print(logistic.score(X, y))
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/098d5a26-ec00-4f5f-ad43-f499daba4c43)
特徵欄位所訓練出的模型準確率約 52.7%，我們可得知結果為一半左右的數值，特徵欄位進行訓練不見得會準確，對於模型的準確度將會有所提升。
### 程式碼：
```python
submit.to_csv( 'Titanic_RandomForest.csv', index=False )
print( f'預測結果：' )
submit
```
### 執行結果：
產生後儲存為資料表後上傳至Kaggle的Submit Predictions，然後按Submit就完成了此競賽項目。
# 三、結論
使用 Kaggle 上鐵達尼號生存預測比賽，使用了性別、票務艙、登船港口、兄弟姊妹配偶人數、父母子女人數與生存率的關係做出相關關資料的分析及處理技巧，也用訓練模型、測試模型來預測、觀察及嘗試，運用在資料科學或機器學習專案作品上會能夠凸顯出學習經驗成訣竅。因此，一個專案要能順利進行且有收穫，除需具備熟練的程式語言外，該領域的專業知識及實務經驗，更是一大關鍵的因素，讓我們一同來學習成長！
# 四、參考
Hahow學習AI一把抓：點亮人工智慧技能樹。

Hahow Python資料分析：AI機器學習入門到應用。

巨匠電腦Python機器學習應用開發。

TQC+ Python3.x 機器學習基礎與應用特訓教材。
