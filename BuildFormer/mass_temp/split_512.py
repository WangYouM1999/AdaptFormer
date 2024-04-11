import numpy as np
import cv2
import os


def pad_mask(img):
    ret = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=1)
    return ret


def crop_mask(mask, split_size, stride, img_name, save_path):
    index = 0
    for y in range(0, mask.shape[0], stride):
        for x in range(0, mask.shape[1], stride):
            mask_tile_cut = mask[y:y + split_size, x:x + split_size]
            cur_name = img_name + '_'+str(index) + ".png"
            cv2.imwrite(save_path + "/" + cur_name, mask_tile_cut)
            index += 1
    print("Total masks:", index)


if __name__ == "__main__":
    path = "/home/wym/projects/BuildFormer/fig_results/massbuilding"
    cnt = 0
    dirs = os.listdir(path)
    print(dirs)
    for dir in os.listdir(path):
        for img_name in os.listdir(dir):
            if img_name.endswith(".png"):  # 只处理.png文件
                pure_name = img_name.split(".")[0]

                mask = cv2.imread(dir + "/" + img_name, cv2.IMREAD_UNCHANGED)

                mask_pad = pad_mask(mask)

                sava_path = os.path.join(path+"/split_512", dir)
                if not os.path.exists(sava_path):
                    os.makedirs(sava_path)
                crop_mask(mask_pad, 512, 512, pure_name, sava_path)
                cnt += 1
        print(cnt)