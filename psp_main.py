import torch
import torch.nn as nn
import torch.optim as optim # Optimization algorithm 모음
import os
import torch.nn.functional as F # 파라미터가 필요없는 Function 모음
import cv2
from psp_function import *
from psp_network import PSPNet
from unet_network import UNet
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Dataset loading...')
path1 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
path2 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train_val.txt"
path3 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages" # train용
path4 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass" # test용

train_name, val_name = txt_load(path1), txt_load(path2)
train_img, val_img = img_load(path3, train_name), img_load(path3, val_name)
train_gt_img, val_gt_img = gt_load(path4, train_name), gt_load(path4, val_name)

labeling_train_gt_img, labeling_val_gt_img = labeling_gt(train_name, train_gt_img), labeling_gt(val_name, val_gt_img)
# np.save("psp_labeling_train_gt.npy", labeling_train_gt_img)
# np.save("psp_labeling_val_gt.npy", labeling_val_gt_img)
# labeling_train_gt_img = np.load("psp_labeling_train_gt.npy")
# labeling_val_gt_img = np.load("psp_labeling_val_gt.npy")
print('Dataset loading complete')

total_miou = np.zeros((21, 21), dtype=int)
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005, eps=1.0, weight_decay=0.0001)

iteration = 150000
with tqdm(range(iteration + 1), dynamic_ncols=True, miniters=1) as pbar:
    for i in pbar:
        model.train()
        new_train_image, new_train_gt_image = minibatch(train_img, labeling_train_gt_img, batch_size=32)
        new_train_image = torch.from_numpy(new_train_image.astype(np.float32)).to(device)

        new_train_gt_image = torch.from_numpy(new_train_gt_image.astype(np.float32)).to(device)
        new_train_gt_image = new_train_gt_image.permute(0, 3, 1, 2)
        new_train_image = new_train_image.permute(0, 3, 1, 2)

        output, aux_output = model(new_train_image)

        aux_output = torch.nn.functional.upsample_bilinear(aux_output, size=(new_train_gt_image.shape[2], new_train_gt_image.shape[3]))
        main_loss = F.cross_entropy(output, new_train_gt_image)
        aux_loss = F.cross_entropy(aux_output, new_train_gt_image)

        total_loss = main_loss + 0.4 * aux_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if 50000 <= i and i < 75000:
            optimizer.param_groups[0]['lr'] = 0.0003
        if 75000 <= i and i < 100000:
            optimizer.param_groups[0]['lr'] = 0.0001
        if 100000 <= i and i < 125000:
            optimizer.param_groups[0]['lr'] = 0.00007
        if 125000 <= i and i < 150000:
            optimizer.param_groups[0]['lr'] = 0.00005

        if i % 100 == 0:
            pbar.set_postfix({"Step": f"{i}/{iteration}", "Loss": f"{total_loss.item():.4f}"})

        if i % 5000 == 0:
            torch.save(model.state_dict(), 'aux_0.0005_pspnet_model_{i}.pt'.format(i=i))
        if i % 5000 == 0 and i != 0:
        # if i % 5000 == 0:
            model.eval()
            correct = 0

            with torch.no_grad():
                for idx in range(len(val_img)):
                    imag = val_img[idx]
                    height, width = imag.shape[0], imag.shape[1]

                    imag_tensor = torch.from_numpy(imag.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
                    imag_tensor = (imag_tensor / 255.0) * 2.0 - 1.0
                    pred = model(imag_tensor)  # 1, num_classes, height, width
                    # pred = F.interpolate(pred, size=(labeling_val_gt_img.shape[1], labeling_val_gt_img.shape[2]), mode='bilinear', align_corners=True)
                    pred = pred.cpu().numpy()

                    pred = pred.transpose(0, 2, 3, 1)
                    pred = pred.squeeze(0)

                    img = seg_to_rgb(pred)

                    checkpoint = f'C:/Users/user/Desktop/semantic segmentation/testing image/PSPNet/{i}'
                    os.makedirs(checkpoint, exist_ok=True)
                    original_image_name = val_name[idx]
                    cv2.imwrite(f'{checkpoint}/testing_image_{original_image_name}.png', img)

                    pred = np.array(pred)
                    gt = np.array(labeling_val_gt_img[idx])
                    func = mIoU(pred, gt)

                    total_miou += (func)

                iray = []
                count = 21

                for m in range(21):
                    union = sum(total_miou[m, :]) + sum(total_miou[:, m]) - total_miou[m, m]
                    intersection = total_miou[m, m]

                    iou = 0 if union == 0 else intersection / union
                    iray.append(iou)

                diag_intersection = np.diag(total_miou)
                pixel_accuracy = sum(diag_intersection) / (256 * 256 * 1449)
                mean_iou = sum(iray) / count
                print("mIoU : {}, pixel_accuracy : {}".format(mean_iou, pixel_accuracy))
                total_miou = np.zeros((21, 21), dtype=int)

                # 로그 기록
                original_image_name = val_name[idx]
                if i <= 50000:
                    filename = 'u50000_mIoU.txt'
                    pixel = 'u50000_pixel.txt'
                elif i <= 100000:
                    filename = 'u100000_mIoU.txt'
                    pixel = 'u100000_pixel.txt'
                elif i <= 150000:
                    filename = 'u150000_mIoU.txt'
                    pixel = 'u150000_pixel.txt'
                else:
                    filename = 'u200000_mIoU.txt'
                    pixel = 'u200000_pixel.txt'

                filepath1 = os.path.join('C:/Users/user/Desktop/semantic segmentation/testing image/Unet', filename)
                filepath2 = os.path.join('C:/Users/user/Desktop/semantic segmentation/testing image/Unet', pixel)
                with open(filepath1, "a+") as tmp:
                    tmp.write(f' i : {i}, mIoU : {mean_iou}\n')
                with open(filepath2, "a+") as tmp2:
                    tmp2.write(f' i : {i}, pixel_accuracy : {pixel_accuracy}\n')
                mean_iou = 0
                pixel_accuracy = 0
                cv2.imwrite(f'{checkpoint}/testing_image_{original_image_name}.png', img)
            
