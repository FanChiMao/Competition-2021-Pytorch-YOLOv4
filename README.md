# Pytorch-YOLOv4-AI CUP  
- [水稻無人機全彩影像植株位置自動標註與應用](https://aidea-web.tw/topic/9c88c428-0aa7-480b-85e0-2d8fb2fcf3fc)  
```
├── README.md    

主要訓練程式碼
├── cfg.py                  訓練相關參數
├── make_txt.py             把主辦單位給的csv轉成相關格式
├── train.txt               轉檔後的訓練標籤檔
├── val.txt                 轉檔後的驗證標籤檔
├── train.py                執行訓練及其他參數調整
├── training                預設訓練及驗證圖片資料夾 
├── log                     訓練loss可視化(tensorboard)
├── checkpoint              儲存每個epoch的權重檔
├── yolov4.conv.137.pth     YOLOv4 pretrained model
├── yolov4-csp.conv.142.pth Scaled YOLOv4 pretrained model   

主要測試程式碼   
├── TestResult.py           測試水稻
├── test_cfg.yaml           設定測試水稻參數      
├── dataset.py              讀檔轉檔
├── classify.py             MLP分類器  
├── TestMask.py             U-Net產生Type0 mask
├── testingType0            產生Type0 mask用
├── testing                 預設測試圖片資料夾
├── tool
│   ├── coco_annotation.py
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
├── TypeX_model             對應不同type的model
│   ├── Best AP_(第x個epoch)x(資料增強倍數).pth
│   ├── ...
├── Result                  測試結果圖及對應csv

```
# 0. 訓練

## 0.1 準備訓練資料  
- 圖片準備根據主辦單位給的[訓練資料](https://drive.google.com/drive/folders/1s_JVoaABFFWzzaZ9U0fIEXQ5z_E5OGEd?usp=sharing)總共43張圖及對應labels (label格式為水稻中心點座標):  
- 利用`make_txt.py`把主辦單位給的水稻中心excel檔轉成YOLO需要的`train.txt` 與 `val.txt`檔  

## 0.2 相關參數設定
1. 準備Ground truth label (`train.txt`/`val.txt`)  
   並將訓練圖片放入training資料夾，label格式如下
    ```
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    ```
2. 設定`cfg.py`參數
- 若選擇darknet當作backbone (`Cfg.use_darknet_cfg = True`)  
  需要先找對應的cfg (`yolov4-csp.cfg`)檔，找到classes部分改為所需類別，以及filters部分  
  filters 的大小為 = (5 + classes)*3
    ```
     ...
     [convolutional]
     size=1
     stride=1
     pad=1
     filters=18 #(39)
     activation=linear
    
    
     [yolo]
     mask = 6,7,8
     anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
     classes=1  #(8)
     ...
    ```   
- 若不用darknet，則需要在`train.py`裡，Yolo_loss中的image_size調整resize大小
    ```   
    class Yolo_loss(nn.Module):
        def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
            super(Yolo_loss, self).__init__()
            self.device = device
            self.strides = [8, 16, 32]
            image_size = 640
            self.n_classes = n_classes
            self.n_anchors = n_anchors
    ```   

3. train
    ```
    python train.py -l [學習率] -g [GPU] -dir [訓練資料集位置] -classes [類別]
    ```
# 1. 測試

## 1.1 相關測試參數設定
1. [AI CUP 競賽報告](https://drive.google.com/file/d/1puLpWeq7S_aKfyerbI9787HfJ-Fl19_l/view?usp=sharing)  
2. [AI CUP 實驗記錄](https://drive.google.com/file/d/1tNn-kyzaWkC-EPw4iEtFYSf3xShvJVQq/view?usp=sharing)  
3. [Public data](https://drive.google.com/drive/folders/1lx4rOFNm1ayZOFxhmhru6AoiEg05JO4O?usp=sharing)
4. [Private data](https://drive.google.com/drive/folders/1n52IcT7IGtNQ5OG2wetj__WAki9ajiRO?usp=sharing)
5. 測試時不需要更改相關路徑，只須確定所有相對路徑是否有圖片  
6. 測試時所有更改參數的地方都在`test_cfg.yaml`進行更改  
7. 預設測試資料路徑: `./testing`
8. 預設測試結果路徑: `./Result`

## 1.2 開始測試  
    
```
python TestResult.py
```

# 2. 測試結果

## 2.1 結果圖
- Type1: `DSC081133.jpg`  
    ![image](https://i.ibb.co/mvW0Skr/DSC081133.jpg)  
  &nbsp;
- Type0: `IMG-170406-035957-0043-RGB2.jpg`  
    ![image](https://i.ibb.co/G77wxj4/IMG-170406-035957-0043-RGB2.jpg)  
  &nbsp;
- Type2: `IMG_170406_040356_0242_RGB4.jpg`  
    ![image](https://i.ibb.co/VLjZf9M/IMG-170406-040356-0242-RGB4.jpg)  
  &nbsp;
## 2.2 測試分數
- 我們每次上傳分數都會留下當次測試的參數細節、偵測結果圖與測試分數  
  (https://drive.google.com/drive/folders/1EZeyRFDi9dmy7UYNcRN06V7a6xarPK2G?usp=sharing)  
  p.s. 整體檔案大小31G 上面分享網址只包含最後一次更新分數計算的每次測試結果資料  
       若有需要可以聯絡我們 再把所有完整檔案分批傳送
  

- Public best config (Copy to `test_cfg.yaml`)  
    ```
    ---
    test:
      model ensemble: true
      image ensemble: true
      gpu: true
      model0_1: './Type0_model/Best AP_629x32.pth'
      model0_2: './Type0_model/Best AP_462x6.pth'
      model0_3: './Type0_model/Best AP_95x32_928.pth'
      model1_1: './Type1_model/Best AP_211_640csp.pth'
      model1_2: './Type1_model/Best AP_500.pth'
      model1_3: './Type1_model/Best AP_57x96.pth'
      model2_1: './Type2_model/Best F1_353.pth'
      model2_2: './Type2_model/Best F1_353.pth'
      model2_3: './Type2_model/Best AP_25x80.pth'
      type0_c_n: [0.1, 0]
      type1_c_n: [0.06, 0.15]
      type2_c_n: [0.5, 0.01]
      area constant0: 200
      area constant1: 1300
      area constant2: 0
      data path: './testing/'
      mask path: './UNetMaskOptimize/'
      classes: 1
      boundary: 0.5
      width: 608
      height: 608
      save img: './Result/'
      save csv: './Result/csv'
    
    
    classifier:
      model path: './classifier/all_right_model.pth'
      name json: './classifier/custom_to_name.json'
    ...
    ```
  &nbsp;
- [Public dataset](https://drive.google.com/drive/folders/1lx4rOFNm1ayZOFxhmhru6AoiEg05JO4O?usp=sharing):  
    - Best score 
    
        |                     |  F1-score    |  Precision   |  Recall      |
        | ------------------- | :----------: | :----------: | :----------: |
        | Best result         |    0.9440449|      0.94246|      0.94759|  
  
    - Leaderboard  
    ![image](https://i.ibb.co/thC0G2p/1624291305349.jpg)  
&nbsp;
- [Private dataset](https://drive.google.com/drive/folders/1n52IcT7IGtNQ5OG2wetj__WAki9ajiRO?usp=sharing):      
    - Best score   
    
        |                     |  F1-score    |  Precision   |  Recall      |
        | ------------------- | :----------: | :----------: | :----------: |
        | Best result         |    0.9354838|      0.91483|      0.96065|  
    
    - Leaderboard  
    ![image](https://i.ibb.co/28qtND6/image.jpg)  
    
&nbsp;
- [官方公告最終排名](https://drive.google.com/file/d/1_1UAY8xM4EQlbQ6mt5_Fqg0Q8dY3sIbP/view?usp=sharing):  

  - 參加隊伍: 523隊  

  - 最終排名: 總排名第 **11** 名 (佳作),  
    全國大專院校組第 **9** 名(原第4、第5為業界隊伍)  

&nbsp;
# 3 參考資料
   
- Reference:
   - https://github.com/eriklindernoren/PyTorch-YOLOv3
   - https://github.com/marvis/pytorch-caffe-darknet-convert
   - https://github.com/marvis/pytorch-yolo3
   - https://github.com/Tianxiaomo/pytorch-YOLOv4
   - https://github.com/milesial/Pytorch-UNet
   - https://github.com/jclh/image-classifier-PyTorch

# 4 聯絡資訊
- E-mail: qaz5517359@gmail.com
