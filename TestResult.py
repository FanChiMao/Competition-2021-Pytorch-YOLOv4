import yaml

import classify
from tool.torch_utils import *


def sharp(img, sigma):
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp_img = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return sharp_img


def ensemble_image(image):
    import cv2
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    # V.fill(255)
    V = np.round(V * 1.5)
    V[V > 255] = 255
    V = V.astype(np.uint8)
    bright = cv2.merge([H, S, V])
    bright = cv2.cvtColor(bright, cv2.COLOR_HSV2RGB)

    H, S, V = cv2.split(HSV)
    # S.fill(255)
    S = np.round(S * 1.5)
    S[S > 255] = 255
    S = S.astype(np.uint8)
    saturation = cv2.merge([H, S, V])
    saturation = cv2.cvtColor(saturation, cv2.COLOR_HSV2RGB)

    B, G, R = cv2.split(image)
    b1 = cv2.equalizeHist(B)
    g1 = cv2.equalizeHist(G)
    r1 = cv2.equalizeHist(R)
    contrast = cv2.merge([b1, g1, r1])
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)

    return original, bright, saturation, contrast


if __name__ == "__main__":
    import os
    import cv2
    from models import load_multi_model
    from tool.utils import plot_boxes_and_create_csv, majority_vote_boxes_set
    from tool.torch_utils import do_detect, do_three_detect

    # Load yaml configuration file
    with open('test_cfg.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    # Declare variable
    test = config['test']
    class_cfg = config['classifier']
    judge_type = classify.load_eval_model(class_cfg['model path'])
    use_cuda = test['gpu']
    path = test['data path']
    img_path = test['save img']
    csv_path = test['save csv']
    a0 = test['area constant0']
    a1 = test['area constant1']
    a2 = test['area constant2']
    b = test['boundary']
    type0_prediction = 0
    type1_prediction = 0
    type2_prediction = 0
    print('---------------------------------------------')
    print(f'''Testing start:
    
    Testing images:    {path}
    Result images:     {img_path}
    Image ensemble:    {test['image ensemble']}
    Model ensemble:    {test['model ensemble']}
    Type0 conf/nms:    {test['type0_c_n']}
    Type1 conf/nms:    {test['type1_c_n']}
    Type2 conf/nms:    {test['type2_c_n']}
    Area constant0:    {a0}
    Area constant1:    {a1}
    Area constant2:    {a2}
    Boundary constant: {b}
    Classes:           {test['classes']}
    Wight:             {test['width']}
    Height:            {test['height']}
    GPU:               {use_cuda}       
        ''')
    print('---------------------------------------------')

    start_time = time.time()
    # model_0 密集小水稻, model_1 密集大水稻, model_2 稀疏大水稻
    # Load model's weights and bias
    model_0_1, model_0_2, model_0_3, model_1_1, model_1_2, model_1_3, model_2_1 = \
        load_multi_model(test['model0_1'], test['model0_2'], test['model0_3'],
                         test['model1_1'], test['model1_2'], test['model1_3'],
                         test['model2_1'])
    count = 1
    # Read the testing data(images)
    allFileList = os.listdir(path)
    for file in allFileList:
        imgfile = path + file  # 存放測試集(圖片)的絕對路徑
        image_name = os.path.splitext(file)[0]  # 測試圖片的名稱: 給輸出csv和圖片的檔名用
        img = cv2.imread(imgfile)
        # Resized the image and generate the contrast, saturation image for image ensemble (RGB)
        sized = cv2.resize(img, (test['width'], test['height']))
        image_original, image_bright, image_saturation, image_contrast = ensemble_image(sized)
        sized928 = cv2.resize(img, (928, 928))
        image_original928 = cv2.cvtColor(sized928, cv2.COLOR_BGR2RGB)
        sized640 = cv2.resize(img, (640, 640))
        image_original640, image_bright640, image_saturation640, image_contrast640 = ensemble_image(sized640)
        # ------分類器(MLP) --------------------------------------------------------------------------------------------
        # classify the type of testing image (vgg16 + perceptron)
        class_to_name_dict = classify.load_json(class_cfg['name json'])
        with torch.no_grad():  # image to classifier
            probabilities, classes = classify.predict(imgfile, judge_type, topk=3, gpu=use_cuda)
        type_number = (int(classes[0]) - 1)  # classes = [機率第一的類別 第二 第三] (1: type0, 2: type1, 3: type2)
        print('Image %d' % count + ' --> (vgg16 + perceptron) --> ' + 'Type %d' % type_number)
        # --------------------------------------------------------------------------------------------------------------
        if type_number == 1:  # type 1
            if test['model ensemble'] and test['image ensemble']:
                boxes_1_1, boxes_1_1_s, boxes_1_1_v = do_three_detect(model_1_1, image_original, image_bright,
                                                                      image_saturation, test['type1_c_n'][0],
                                                                      test['type1_c_n'][1], use_cuda)
                boxes_1_2, boxes_1_2_s, boxes_1_2_v = do_three_detect(model_1_2, image_original640, image_bright640,
                                                                      image_saturation640, test['type1_c_n'][0],
                                                                      test['type1_c_n'][1], use_cuda)
                boxes_1_3, boxes_1_3_s, boxes_1_3_v = do_three_detect(model_1_3, image_original, image_bright,
                                                                      image_saturation, test['type1_c_n'][0],
                                                                      test['type1_c_n'][1], use_cuda)
                print('Image ensemble:', end='')
                vote_box_1 = majority_vote_boxes_set(img, boxes_1_1[0], boxes_1_1_s[0], boxes_1_1_v[0])
                print('Image ensemble:', end='')
                vote_box_2 = majority_vote_boxes_set(img, boxes_1_2[0], boxes_1_2_s[0], boxes_1_2_v[0])
                print('Image ensemble:', end='')
                vote_box_3 = majority_vote_boxes_set(img, boxes_1_3[0], boxes_1_3_s[0], boxes_1_3_v[0])
                print('Model ensemble:', end='')
                Final_boxes = majority_vote_boxes_set(img, vote_box_1, vote_box_2, vote_box_3)
                number1 = plot_boxes_and_create_csv(img, Final_boxes, image_name, img_path, csv_path, SetBoundary=b,
                                                    SetArea=a1)
            elif test['model ensemble']:  # no image ensemble
                boxes_1_1 = do_detect(model_1_1, image_original, test['type1_c_n'][0], test['type1_c_n'][1], use_cuda)
                boxes_1_2 = do_detect(model_1_2, image_original, test['type1_c_n'][0], test['type1_c_n'][1], use_cuda)
                boxes_1_3 = do_detect(model_1_3, image_original, test['type1_c_n'][0], test['type1_c_n'][1], use_cuda)
                print('Model ensemble:', end='')
                boxes_1_1 = do_detect(model_1_1, image_original, test['type1_c_n'][0], test['type1_c_n'][1], use_cuda)
                New_box1 = majority_vote_boxes_set(img, boxes_1_1[0], boxes_1_2[0], boxes_1_3[0])
                number1 = plot_boxes_and_create_csv(img, New_box1, image_name, img_path, csv_path, SetBoundary=b,
                                                    SetArea=a1)
            else:
                boxes_1_1 = do_detect(model_1_1, image_original640, test['type1_c_n'][0], test['type1_c_n'][1],
                                      use_cuda)
                number1 = plot_boxes_and_create_csv(img, boxes_1_1[0], image_name, img_path, csv_path, SetBoundary=b,
                                                    SetArea=a1)
            type1_prediction += number1

        elif type_number == 0:  # type 0
            # process the mask
            try:
                final_path = test['mask path']
                mask = cv2.imread(final_path + file)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            except:
                mask = None

            if test['model ensemble']:  # model ensemble
                print('Model ensemble:', end='')
                boxes_0_1 = do_detect(model_0_1, image_original, test['type0_c_n'][0], test['type0_c_n'][1], use_cuda)
                boxes_0_2 = do_detect(model_0_2, image_original, test['type0_c_n'][0], test['type0_c_n'][1], use_cuda)
                boxes_0_3 = do_detect(model_0_3, image_original928, test['type0_c_n'][0], test['type0_c_n'][1],
                                      use_cuda)
                New_box0 = majority_vote_boxes_set(img, boxes_0_1[0], boxes_0_2[0], boxes_0_3[0])
                number0 = plot_boxes_and_create_csv(img, New_box0, image_name, img_path, csv_path, SetBoundary=b,
                                                    SetArea=a0, mask=mask)
            else:  # single model
                boxes_0_1 = do_detect(model_0_1, image_original, test['type0_c_n'][0], test['type0_c_n'][1], use_cuda)
                number0 = plot_boxes_and_create_csv(img, boxes_0_1[0], image_name, img_path, csv_path, SetBoundary=b,
                                                    SetArea=a0, mask=mask)
            type0_prediction += number0

        elif type_number == 2:  # type 2
            if test['image ensemble']:
                print('Image ensemble:', end='')
                boxes_2_1, boxes_2_1_s, boxes_2_1_v = do_three_detect(model_2_1, image_original, image_bright,
                                                                      image_saturation, test['type2_c_n'][0],
                                                                      test['type2_c_n'][1], use_cuda)
                vote_box_21 = majority_vote_boxes_set(img, boxes_2_1[0], boxes_2_1_s[0], boxes_2_1_v[0])
                number2 = plot_boxes_and_create_csv(img, vote_box_21, image_name, img_path, csv_path, SetBoundary=b,
                                                    SetArea=a2)
            else:
                boxes_2_1 = do_detect(model_2_1, image_original, test['type2_c_n'][0], test['type2_c_n'][1], use_cuda)
                number2 = plot_boxes_and_create_csv(img, boxes_2_1[0], image_name, img_path, csv_path, SetBoundary=b,
                                                    SetArea=a2)

            type2_prediction += number2
        total_prediction = (type0_prediction + type1_prediction + type2_prediction)
        print('Save image ' + file)
        print(f'                                      ({count}/{len(allFileList)})')
        print('---------------------------------------------')
        count += 1
    finish_time = time.time()
    process_min = int((finish_time - start_time) / 60)
    process_sec = int((finish_time - start_time) - process_min * 60)
    print(f'''Testing finish:

    Total processing time:    {process_min} min {process_sec} seconds
    Type0 prediction number:  {type0_prediction}
    Type1 prediction number:  {type1_prediction}
    Type2 prediction number:  {type2_prediction}
    Total prediction number:  {total_prediction}   
    
    ''')
    print('---------------------------------------------')
