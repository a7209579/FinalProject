# 基於GaborNet改善人臉表情辨識
## 摘要
本文中我們使用GaborNet卷積神經網路模型，改善傳統先使用Gabor Filter提取特徵再放入CNN模型辦識的方法。GaborNet具有可學習Gabor函式參數的Gabor層，利用反向傳播演算法更新每輪訓練的參数。GaborNet模型比對使用傳統方法的人臉表情辩識論文，實驗結果顯示，此方法步驟更少，只需一層Gabor層和一層卷積層就可以達到更好的辦識率。
## 資料集
* [The Japanese Female Facial Expression (JAFFE) Dataset](https://zenodo.org/record/3451524#.YXe3sC9Cb0o)

| category |  AN | DI | FE | HA | SA | SU | NE | Total |
|----------|:---:|---:|---:|---:|---:|---:|---:|------:|
|**images**|  30 | 29 | 32 | 31 | 31 | 30 | 30 |**213**|


## 模型架構
使用一層Gabor層和一層卷積層的CNN模型做為JAFFE資料集辨識的類神經網路架構。運用一層Gabor層和一層卷積層，並且每層做2x2的MaxPooling，ReLU作為Gabor層和卷積層的啟動函數，Softmax則作為最後全連接層的啟動函數。Gabor層中的Gabor kernel做為CNN的kernel，kernel大小設為9x9。
![image](https://github.com/a7209579/FinalYearProject/blob/main/images/structure.png)
## 開發環境
#### 硬體
* CPU: Intel i7-11700KF
* GPU: NVIDIA GeForce RTX3070
#### 開發軟體
* python 3.8.9
#### 使用套件
```
pip install torch 1.9.0+cu111
```
```
pip installtorchvision1.9.0+cu111
```
pip installmatplotlib 3.4.3
pip installseaborn 0.11.2
pip installscikit-image 0.18.3

## 實驗結果
* ![image](https://github.com/a7209579/FinalYearProject/blob/main/images/acc.png)
* ![image](https://github.com/a7209579/FinalYearProject/blob/main/images/confusion_matrix.png)
## 參考文獻
test
