import os
import glob
import pandas as pd
import cv2
import csv
from PIL import Image


def readandwrite_csv(data_txt=None, data_csv=None, name=None, wh=None, max_img=None):
    data_txt.write(str(name) + ' ')  # 檔名 + 空格
    row_number = 1
    f = open(data_csv)
    max_num = len(f.readlines())
    f.close()
    with open(data_csv) as csvFile:
        rows = csv.reader(csvFile)
        for row in rows:  # csv檔所有的點
            x1 = int(row[0]) - wh  # 左上x
            y1 = int(row[1]) - wh  # 左上y
            x2 = int(row[0]) + wh  # 右下x
            y2 = int(row[1]) + wh  # 右下y
            # 0是類別

            if row_number == max_num:
                data_txt.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(0))
            else:
                data_txt.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(0) + ' ')
            row_number += 1
        if name == max_img:
            pass
        else:
            data_txt.write('\n')  # 換行


# 訓練集資料夾
# |-- 訓練圖片資料夾 (train)
# |   |-- image 1.jpg
# |   |-- image 1.csv
# |   |-- image 2.jpg
# |   |-- image 2.csv
# |   |...
# |   |-- 驗證圖片資料夾 (validation)
# |   |   |-- image 1.jpg
# |   |   |-- image 1.csv
# |   |   |-- image 2.jpg
# |   |   |-- image 2.csv
# |   |   |-- ...
# GT image path & csv
train_path = 'D:/NCHU/1092/AICUP/Dataset/train_txt_type0/'  # 訓練圖片資料夾路徑
val_path = train_path + 'val/'  # 驗證圖片資料夾路徑
save_img_path = './training/'  # 儲存圖片資料夾路徑
train_imgFileList = sorted(glob.glob(train_path + '*.jpg'))
train_csvFileList = sorted(glob.glob(train_path + '*.csv'))
train_f = open('train.txt', 'w')
val_imgFileList = sorted(glob.glob(val_path + '*.jpg'))
val_csvFileList = sorted(glob.glob(val_path + '*.csv'))
val_f = open('val.txt', 'w')
count = 0  # 檔名
boundary = 10  # boundary邊長

if __name__ == "__main__":
    train_number = len(train_imgFileList)
    print('---train')
    # train.txt-------------------------------------------------------------------------
    for (train_img, train_csv) in zip(train_imgFileList, train_csvFileList):
        image = cv2.imread(train_img)
        count += 1
        readandwrite_csv(train_f, train_csv, count, boundary, train_number)
        cv2.imwrite(save_img_path + str(count) + '.jpg', image)  # 儲存圖片
        print(str(count) + '.jpg')
    print('---validation')

    # val.txt-------------------------------------------------------------------------
    val_number = len(val_imgFileList)
    count = train_number
    for (val_img, val_csv) in zip(val_imgFileList, val_csvFileList):
        image1 = cv2.imread(val_img)
        count += 1
        readandwrite_csv(val_f, val_csv, count, boundary, train_number + val_number)
        cv2.imwrite(save_img_path + str(count) + '.jpg', image1)  # 儲存圖片
        print(str(count) + '.jpg')

    train_f.close()
    val_f.close()
    print('---finish!')
