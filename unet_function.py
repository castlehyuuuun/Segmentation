import numpy as np
import os
import cv2
import torch



def loader_train(path, name):
    train_path = path + '/{}.txt'.format(name)

    with open(train_path, 'r') as f:
        lines = f.readlines()
    train = [line.strip() for line in lines]

    train = sorted(train)

    return train

def loader_test(path, name):
    val_path = path + '/{}_val.txt'.format(name)

    with open(val_path, 'r') as f:
       lines = f.readlines()
    val = [line.strip() for line in lines]

    val = sorted(val)

    return val

def loader2_train(path, train_image, name):
    img_path = path + '/{}'.format(name)

    files = os.listdir(img_path)

    files = sorted(files)

    num_images = len(files)
    img = []
    namae = []

    for it in range(num_images):
        file_base = files[it].split('.jpg')[0]
        if file_base in train_image:
            path_tmp = os.path.join(img_path, files[it])
            t = cv2.imread(path_tmp)

            t_resized = cv2.resize(t, (256, 256), interpolation=cv2.INTER_NEAREST)
            img.append(t_resized)
            namae.append(file_base)
        if it % 1000 == 0:
            print("train image loading : {} / {}".format(it, num_images))

    img = np.array(img)
    return namae, img

def loader2_val(path, val_image, name):
    img_path = path + '/{}'.format(name)

    files = os.listdir(img_path)

    files = sorted(files)

    num_images = len(files)
    img = []
    nawa = []

    for it in range(num_images):
        file_base = files[it].split('.jpg')[0]
        if file_base in val_image:
            path_tmp = os.path.join(img_path, files[it])
            t = cv2.imread(path_tmp)

            t_resized = cv2.resize(t, (256, 256), interpolation=cv2.INTER_NEAREST)
            img.append(t_resized)
            nawa.append(file_base)

        if it % 1000 == 0:
            print("val image loading : {} / {}".format(it, num_images))

    img = np.array(img)
    return nawa, img

def loader_gt_train(path, new_train_name, name):
    img_path = path + '/{}'.format(name)

    files = os.listdir(img_path)
    files = sorted(files)

    valid_files = [file for file in files if file.split('.png')[0] in new_train_name]
    num_valid_images = len(valid_files)
    img = np.zeros((num_valid_images, 256, 256, 3), dtype=np.uint8)
    namae = []

    count = 0
    for file in files:
        file_base = file.split('.png')[0]
        if file_base in new_train_name:
            path_tmp = os.path.join(img_path, file)
            t = cv2.imread(path_tmp)

            t_resized = cv2.resize(t, (256, 256), interpolation=cv2.INTER_NEAREST)

            img[count, :, :, :] = t_resized
            namae.append(file_base)
            count += 1

            if count % 100 == 0:
                print("train gt image loading : {} / {}".format(count, num_valid_images))

    return namae, img

def loader_gt_val(path, new_val_name, name):
    img_path = path + '/{}'.format(name)

    files = os.listdir(img_path)
    files = sorted(files)

    valid_files = [file for file in files if file.split('.png')[0] in new_val_name]
    num_valid_images = len(valid_files)
    img = np.zeros((num_valid_images, 256, 256, 3), dtype=np.uint8)
    nawa = []

    count = 0
    for file in files:
        file_base = file.split('.png')[0]
        if file_base in new_val_name:
            path_tmp = os.path.join(img_path, file)
            t = cv2.imread(path_tmp)

            t_resized = cv2.resize(t, (256, 256), interpolation=cv2.INTER_NEAREST)

            img[count, :, :, :] = t_resized
            nawa.append(file_base)
            count += 1

            if count % 100 == 0:
                print("val gt image loading : {} / {}".format(count, num_valid_images))


    return nawa, img


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

# VOC_DICT = {tuple(VOC_COLORMAP[i]): VOC_CLASSES[i] for i in range(len(VOC_COLORMAP))}


a = {}
for i in range(len(VOC_COLORMAP)):
    a[tuple(VOC_COLORMAP[i])] = VOC_CLASSES[i]

voc_dict = {tuple(VOC_COLORMAP[i]): VOC_CLASSES[i] for i in range(len(VOC_COLORMAP))}

def comp(gt_name, rergb, num_classes=21):
    img = np.zeros((len(gt_name), 256, 256, num_classes), dtype=np.uint8)

    for idx in range(len(gt_name)):
        imag = cv2.resize(rergb[idx], (256, 256), interpolation=cv2.INTER_NEAREST)
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

        mini_img[i, :, :, :] = img_tensor
        mini_img_gt[i, :, :, :] = gt_img_tensor

    return mini_img, mini_img_gt


def mIoU(pred, gt):
    category = np.zeros((21, 21), dtype=int)

    for ih in range(256):
        for iw in range(256):
            i = int(np.argmax(pred[ih, iw]))
            w = int(np.argmax(gt[ih, iw]))

            category[i, w] += 1

    iray = []
    count = 0

    for m in range(21):
        iou = 0
        union = sum(category[m, :]) + sum(category[:, m]) - category[m, m]
        intersection = category[m, m]
        if union == 0:
            iou = 0
        elif category[:, m].any() != 0:
            count += 1
            iou = intersection / union

        iray.append(iou)

    iray = np.array(iray)
    mean_iou = sum(iray) / count

    return mean_iou

def seg_to_rgb(output, nc=21):
    # output = torch.tensor(output, dtype=torch.float32)
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

# using_val_gt_image = np.load("256_val_gt_image .npy")
# img = seg_to_rgb(using_val_gt_image[0])
# cv2.imshow("img", img)
# cv2.waitKey(-1)