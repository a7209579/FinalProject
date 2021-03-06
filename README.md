# 使用GaborNet應用於人臉表情辨識
## 1. 摘要
本文中使用GaborNet卷積神經網路模型中的Gabor卷積層來設計我們的網路模型,應用於人臉表情辨識。Gabor層具有可自動學習Gabor函式參數的能力，不必自己手動設計 Gabor 函式的參數，機器
學習出的參數還能比人工手動設計的好。我們的實驗結果比對先使用手動設計的Gabor filters萃取特徵再放入 CNN 模型辨識的論文。結果顯示，我們的方法不但步驟比較少，只需一層Gabor層和一層卷積層就可以達到更好的準確率，於是可以證明，可自動學習參數的Gabor CNN，比使用手動設計的Gabor filters來輔助CNN的方法還好。

## 2. 資料集
* [The Japanese Female Facial Expression (JAFFE) Dataset](https://zenodo.org/record/3451524#.YXe3sC9Cb0o)

| category |  AN | DI | FE | HA | SA | SU | NE | Total |
|----------|:---:|---:|---:|---:|---:|---:|---:|------:|
|**images**|  30 | 29 | 32 | 31 | 31 | 30 | 30 |**213**|

## 3. 開發環境
### 3-1. 硬體
* CPU: Intel i7-11700KF
* GPU: NVIDIA GeForce RTX3070
### 3-2. 開發軟體
* python 3.8.9
### 3-3. 使用套件
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
```
pip install scikit-image==0.18.3
```
```
pip install matplotlib==3.4.3
```
```
pip install seaborn==0.11.2
```

## 4. 模型架構
使用一層Gabor層和一層卷積層的CNN模型做為JAFFE資料集辨識的類神經網路架構。運用一層Gabor層和一層卷積層，並且每層做2x2的MaxPooling，ReLU作為Gabor層和卷積層的啟動函數，Softmax則作為最後全連接層的啟動函數。Gabor層中的Gabor kernel做為CNN的kernel，kernel大小設為9x9。
![image](https://github.com/a7209579/FinalYearProject/blob/main/images/structure.png)
### 4-1. Summary
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
GaborNN                                  --                        --
├─GaborConv2d: 1-1                       [16, 32, 40, 40]          2,720
├─Conv2d: 1-2                            [16, 64, 18, 18]          18,496
├─Linear: 1-3                            [16, 1024]                5,309,440
├─Linear: 1-4                            [16, 7]                   7,175
==========================================================================================
Total params: 5,337,831
Trainable params: 5,337,831
Non-trainable params: 0
Total mult-adds (M): 247.30
==========================================================================================
Input size (MB): 0.15
Forward/backward pass size (MB): 9.34
Params size (MB): 21.35
Estimated Total Size (MB): 30.84
==========================================================================================
```
## 5. 實驗結果
實驗比較這篇論文的方法：
* [Fast Facial emotion recognition Using Convolutional Neural Networks and Gabor Filters](https://www.researchgate.net/publication/344190368_Fast_Facial_emotion_recognition_Using_Convolutional_Neural_Networks_and_Gabor_Filters/link/5f9a4a7992851c14bcf08802/download)  
* 以下為本實驗的結果:
```
[1]  train acc: 0.0986 train loss: 0.2836
[2]  train acc: 0.1455 train loss: 0.2831
[3]  train acc: 0.1784 train loss: 0.2825
[4]  train acc: 0.1972 train loss: 0.2804
[5]  train acc: 0.2911 train loss: 0.2739
[6]  train acc: 0.3146 train loss: 0.2685
[7]  train_acc: 0.3521 train_loss: 0.2622
[8]  train acc: 0.4413 train_loss: 0.2566
[9]  train_acc: 0.4695 train_loss: 0.2484
[10] train_acc: 0.5164 train_loss: 0.2428
[11] train acc: 0.5164 train loss: 0.2399
[12] train_acc: 0.6056 train loss: 0.2299
[13] train_acc: 0.6948 train_loss: 0.2237
[14] train_acc: 0.7324 train_loss: 0.2149
[15] train acc: 0.7136 train loss: 0.2147
[16] train_acc: 0.8169 train_loss: 0.2046
[17] train_acc: 0.8357 train loss: 0.2012
[18] train_acc: 0.8685 train_loss: 0.1957
[19] train_acc: 0.8638 train loss: 0.1953
[20] train_acc: 0.8826 train_loss: 0.1939
[21] train_acc: 0.8779 train loss: 0.1911
[22] train_acc: 0.9108 train_loss: 0.1869
[23] train acc: 0.9014 train loss: 0.1887
[24] train_acc: 0.9531 train_loss: 0.1825
[25] train_acc: 0.9249 train_loss: 0.1828
[26] train_acc: 0.9296 train_loss: 0.1840
[27] train_acc: 0.9484 train_loss: 0.1818
[28] train_acc: 0.9296 train_loss: 0.1802
[29] train_acc: 0.9577 train_loss: 0.1780
[30] train_acc: 0.9718 train_loss: 0.1757
Finished Training
```
### 5-1. 訓練結果視覺化
對比[這篇論文](https://www.researchgate.net/publication/344190368_Fast_Facial_emotion_recognition_Using_Convolutional_Neural_Networks_and_Gabor_Filters/link/5f9a4a7992851c14bcf08802/download)的方法，我們使用Gabor CNN模型，JAFFE資料集訓練準確率從0.9116提升到0.9718，準確率提升了6%，得以證明，使用一層可學習Gabor filter函式參數特徵的Gabor層，比進行兩次手動調整參數進行特徵提取的效果還好。  
![image](https://github.com/a7209579/FinalYearProject/blob/main/images/acc.png)
### 5-2. 混淆矩陣
Gabor CNN模型在JAFFE資料集辨識的混淆矩陣圖，以下七種標籤AN、DI、FE、HA、SA、SU、NE分別為生氣、厭惡、恐懼、開心、悲傷、驚訝、中性。其中悲傷在所有分類中表現不如其他表情好，只有90%的準確率，其它標籤準確率都在97%的水準之上。  
![image](https://github.com/a7209579/FinalYearProject/blob/main/images/confusion_matrix.png)
### 5-3. 濾波器視覺化
Gabor CNN模型訓練後，Gabor層自己學習訓練出的filters如下圖，此Gabor filter為32x9x9的大小，具有各種不同方向，頻率的Gabor filters。  
![image](https://github.com/a7209579/FinalYearProject/blob/main/images/filter.png)
### 5-4. 特徵圖視覺化
經過上圖的濾波器做卷積後特徵圖結果如下圖，產生32張40x40的特徵圖。  
![image](https://github.com/a7209579/FinalYearProject/blob/main/images/feature_maps.png)
## 6. 參考文獻
* [GaborNet: Gabor filters with learnable parameters in deep convolutional neural networks](https://arxiv.org/pdf/1904.13204.pdf)  
* [GaborNet作者Github專案](https://github.com/iKintosh/GaborNet)  
* [Fast Facial emotion recognition Using Convolutional Neural Networks and Gabor Filters](https://www.researchgate.net/publication/344190368_Fast_Facial_emotion_recognition_Using_Convolutional_Neural_Networks_and_Gabor_Filters/link/5f9a4a7992851c14bcf08802/download)  

