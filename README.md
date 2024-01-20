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
