import torch
from torch import nn
from unet import UNet
from c_utils.data_vis import plot_img_and_mask
from c_utils.dataset import BasicDataset
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def fillHole(im_in):
    # Load image, convert to grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(im_in)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)  # (3,3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Filter using contour area and remove small noise
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5500:  # 5500
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    # Morph close and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # MORPH_RECT, (3,3)
    im_out = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    return im_out


if __name__ == "__main__":
    import os
    from PIL import Image
    from scipy import ndimage

    net = UNet(n_channels=3, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    # net.load_state_dict(torch.load('./UNetModel.pth', map_location=device))
    net.load_state_dict(torch.load('./UNetBestValModel.pth', map_location=device))

    print('               Start generating mask!                ')
    print('-----------------------------------------------------')
    path = './testingType0/'
    img_path = './UNetMask/'
    final_path = './UNetMaskOptimize/'
    result_path = './UNetMaskCombination/'

    count = 1
    allFileList = os.listdir(path)
    for file in allFileList:
        print(file)
        imgfile = path + file
        print(imgfile)
        img = Image.open(imgfile)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1,
                           out_threshold=0.5,
                           device=device)
        # -- Mask generation finish ----------------------------
        print('Generating mask by U-Net')
        result_mask = mask_to_image(mask)
        result_mask.save(img_path + file)

        # -- Start optimize and store the mask ---------------------------
        print('Optimizing the initial mask')
        # Remove small white regions
        result_mask = cv2.imread(img_path + file, cv2.COLOR_RGB2BGR)
        # Remove small black hole
        final_mask = fillHole(img_path + file)
        cv2.imwrite(final_path + file, final_mask)
        print('Save final mask images %d ' % count)

        # -- Element-wise mask and original image ---------------------------
        original = cv2.imread(imgfile, cv2.COLOR_RGB2BGR)
        B, G, R = cv2.split(original)
        final_mask_255 = cv2.imread(final_path + file, cv2.THRESH_BINARY)
        final_mask_binary = final_mask_255.astype(float) / 255
        final_image_B = final_mask_binary * B
        final_image_G = final_mask_binary * G
        final_image_R = final_mask_binary * R
        final_image = cv2.merge([final_image_B, final_image_G, final_image_R])
        # final_image[final_mask_255 <= 20] = [101, 160, 188]  # BGR
        # final_image[final_mask_255 <= 20] = [255, 255, 255]  # BGR

        cv2.imwrite(result_path + file, final_image)
        print('Save final images %d ' % count)
        print('-----------------------------------------------------')
        count += 1

        # -- Turn black mask to red
