# Titanic-Survival-Prediction
# 一、說明
本專題透過Kaggle用鐵達尼號生存預測比賽，使用探索性數據分析認識資料到特徵工程處理欄位的生存率、產生訓練集與測試集等產生模型預測資料。
此專案研究用意，對於人工智慧與大數據分析初學者建立程度外，使用pyton程式碼及相關套件實作，然後結合Kaggle 討論區中的文章，更深入來優化整個分析過程。
# 二、實作(請使用python3.6版做執行)
首先透過Kaggle資料競賽網站，[下載鐵達尼號資料集](https://www.kaggle.com/c/titanic/data)。使用pandas 匯入訓練集與測試集資料，並利用 shape 得知資料的維度且合併。
### 1.	讀取資料合併：
訓練集有 891 筆資料、12 個特徵欄位； 測試集有 418 筆資料、11 個特徵欄位； 其中，訓練集較測試集多了判別乘客罹難或生還的特徵欄位 Survived，0 表示罹難者，1 表示生還者。
### 程式碼：
```python
# 合併train及test的資料 
df_data = df_train.append( df_test )
df_data
```
### 執行結果：
![image](https://github.com/LonelyCaesar/-Titanic-Survival-Prediction/assets/101235367/1f59132d-6a20-4f96-886f-a4bf54cef8ee)

合併後訓練測試集總共有1309筆資料、12個特徵欄位，做出一致性的預測分析及模型訓練就會比較快速好理解。
