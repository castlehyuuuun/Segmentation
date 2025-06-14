import numpy as np
import os
import cv2
import torch

def txt_load(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    txt_name = [line.strip() for line in lines]

    return txt_name

def img_load(path, txt_name):
    files = os.listdir(path)
    num_images = len(files)
    img = []

    for i in range(num_images):
        file = files[i].split('.jpg')[0]
        if file in txt_name:
            path_tmp = os.path.join(path, files[i])
            t = cv2.imread(path_tmp)
            t_resized = cv2.resize(t, (256, 256), interpolation=cv2.INTER_NEAREST)
            img.append(t_resized)

        if i % 1000 == 0:
            print("image loading : {} / {}".format(i, num_images))
    img = np.array(img)

    return img

def gt_load(path, txt_name):
    files = os.listdir(path)

    valid_files = [file for file in files if file.split('.png')[0] in txt_name]
    num_valid_images = len(valid_files)
    img = np.zeros((num_valid_images, 256, 256, 3), dtype=np.uint8)

    count = 0
    for file in files:
        file_base = file.split('.png')[0]
        if file_base in txt_name:
            path_tmp = os.path.join(path, file)
            t = cv2.imread(path_tmp)
            t_resized = cv2.resize(t, (256, 256), interpolation=cv2.INTER_NEAREST)
            img[count, :, :, :] = t_resized
            count += 1

            if count % 100 == 0:
                print("Gt image loading : {} / {}".format(count, num_valid_images))

    return img

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

voc_dict = {tuple(VOC_COLORMAP[i]): VOC_CLASSES[i] for i in range(len(VOC_COLORMAP))}

def labeling_gt(gt_name, gt_img, num_classes=21):
    img = np.zeros((len(gt_name), 256, 256, num_classes), dtype=np.uint8)

    for idx in range(len(gt_name)):
        imag = cv2.resize(gt_img[idx], (256, 256), interpolation=cv2.INTER_NEAREST)
        imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
        height, width = imag.shape[0], imag.shape[1]

        for x in range(height):
            for z in range(width):
                color = tuple(imag[x, z, :])
                if color in voc_dict:
                    class_name = voc_dict[color]
                    class_idx = VOC_CLASSES.index(class_name)
                    img[idx, x, z, class_idx] = 1

        if idx % 100 == 0:
            print("iteration : {}".format(idx))

    return img

def minibatch(img, gt, batch_size=32):
    mini_img = np.zeros((batch_size, 256, 256, 3))
    mini_img_gt = np.zeros((batch_size, 256, 256, 21))

    for i in range(batch_size):
        k = np.random.randint(0, len(img))
        img_tensor = img[k]
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor * 2.0 - 1.0

        gt_img_tensor = gt[k]

        p = np.random.randint(2)
        if p == 1:
            img_tensor = cv2.flip(img_tensor, 1)
            gt_img_tensor = cv2.flip(gt_img_tensor, 1)
        t = np.random.randint(2)
        if t == 1:
            img_tensor = cv2.rotate(img_tensor, cv2.ROTATE_90_CLOCKWISE)
            gt_img_tensor = np.rot90(gt_img_tensor, k=1, axes=(0, 1))

        mini_img[i, :, :, :] = img_tensor
        mini_img_gt[i, :, :, :] = gt_img_tensor

    return mini_img, mini_img_gt

def seg_to_rgb(output):
    output = torch.softmax(torch.tensor(output), dim=-1).numpy()

    label_colors = np.array(
        [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
         [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128],
         [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
         [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192],
         [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128],
         [128, 64, 0]])

    height, width, _ = output.shape
    rgb_output = np.zeros((height, width, 3), dtype=np.uint8)

    for h in range(height):
        for w in range(width):
            hc = np.argmax(output[h, w, :])
            rgb_output[h, w, :] = label_colors[hc]

    return rgb_output

def mIoU(pred, gt):
    category = np.zeros((21, 21), dtype=int)

    for ih in range(256):
        for iw in range(256):
            i = int(np.argmax(pred[ih, iw]))
            w = int(np.argmax(gt[ih, iw]))

            category[i, w] += 1

    return category

