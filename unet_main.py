# # from unet_network import *
# from network import *
# from unet_function import *
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import torch.nn.functional as F
# import cv2
# from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim # Optimization algorithm 모음
import os
import torch.nn.functional as F # 파라미터가 필요없는 Function 모음
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from psp_network import *
from psp_function import *
from unet_network import UNet
from tqdm import tqdm

'''
# cuda가 사용 가능한 지 확인
torch.cuda.is_available()

# cuda가 사용 가능하면 device에 "cuda"를 저장하고 사용 가능하지 않으면 "cpu"를 저장한다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 멀티 GPU 사용 시 사용 가능한 GPU 셋팅 관련
# 아래 코드의 "0,1,2"는 GPU가 3개 있고 그 번호가 0, 1, 2 인 상황의 예제입니다.
# 만약 GPU가 5개이고 사용 가능한 것이 0, 3, 4 라면 "0,3,4" 라고 적으면 됩니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# 현재 PC의 사용가능한 GPU 사용 갯수 확인
torch.cuda.device_count()

# 사용 가능한 device 갯수에 맞춰서 0번 부터 GPU 할당
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, list(range(torch.cuda.device_count())))))

# 실제 사용할 GPU만 선택하려면 아래와 같이 입력하면 됩니다. (예시)
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 4, 6"

# cudnn을 사용하도록 설정. GPU를 사용하고 있으면 기본값은 True 입니다.
import torch.backends.cudnn as cudnn
cudnn.enabled = True

# inbuilt cudnn auto-tuner가 사용 중인 hardware에 가장 적합한 알고리즘을 선택하도록 허용합니다.
https://gaussian37.github.io/dl-pytorch-snippets/ 
위 게시글 참고할 것
cudnn.benchmark = True
'''
print('Dataset loading...')
path1 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
path2 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train_val.txt"
path3 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages" # train용
path4 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass" # test용

train_name, val_name = txt_load(path1), txt_load(path2)
train_img, val_img = img_load(path3, train_name), img_load(path3, val_name)
train_gt_img, val_gt_img = gt_load(path4, train_name), gt_load(path4, val_name)

# labeling_train_gt_img, labeling_val_gt_img = labeling_gt(train_name, train_gt_img), labeling_gt(val_name, val_gt_img)
# np.save("psp_labeling_train_gt.npy", labeling_train_gt_img)
# np.save("psp_labeling_val_gt.npy", labeling_val_gt_img)
labeling_train_gt_img = np.load("psp_labeling_train_gt.npy")
labeling_val_gt_img = np.load("psp_labeling_val_gt.npy")
print('Dataset loading complete')

total_miou = np.zeros((21, 21), dtype=int)
model = UNet().to(device)
# 저장된 모델 가중치 불러오기
# model.load_state_dict(torch.load('wo_bn_relu_PSPNet_model.pt'))
# model.load_state_dict(torch.load('pspnet_model_155001.pt'))
# model.load_state_dict(torch.load('aux_0.005_pspnet_model_30001.pt'))
# model.load_state_dict(torch.load('aux_0.001_pspnet_model_25000.pt'))
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

        # output = model(new_train_image).to(device)
        # # output의 크기를 new_train_gt_image와 맞추기 위해 업샘플링
        # output = F.interpolate(output, size=(new_train_gt_image.shape[2], new_train_gt_image.shape[3]), mode='bilinear', align_corners=True)
        # loss = F.cross_entropy(output, new_train_gt_image)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        output = model(new_train_image)

        # output의 크기를 new_train_gt_image와 맞추기 위해 업샘플링
        # output = F.interpolate(output, size=(new_train_gt_image.shape[2], new_train_gt_image.shape[3]), mode='bilinear', align_corners=True)
        # aux_output = F.interpolate(aux_output, size=(new_train_gt_image.shape[2], new_train_gt_image.shape[3]), mode='bilinear', align_corners=True)
        # aux_output = torch.nn.functional.upsample_bilinear(aux_output, size=(new_train_gt_image.shape[2], new_train_gt_image.shape[3]))
        main_loss = F.cross_entropy(output, new_train_gt_image)
        # aux_loss = F.cross_entropy(aux_output, new_train_gt_image)

        # total_loss = main_loss + 0.4 * aux_loss
        main_loss.backward()
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
            pbar.set_postfix({"Step": f"{i}/{iteration}", "Loss": f"{main_loss.item():.4f}"})

        if i % 5000 == 0:
            torch.save(model.state_dict(), 'aux_0.0005_Unet_model_{i}.pt'.format(i=i))
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

                    checkpoint = f'C:/Users/user/Desktop/semantic segmentation/testing image/UNet/{i}'
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
                    filename = 'U50000_mIoU.txt'
                    pixel = 'U50000_pixel.txt'
                elif i <= 100000:
                    filename = 'U100000_mIoU.txt'
                    pixel = 'U100000_pixel.txt'
                elif i <= 150000:
                    filename = 'U150000_mIoU.txt'
                    pixel = 'U150000_pixel.txt'
                else:
                    filename = 'U200000_mIoU.txt'
                    pixel = 'U200000_pixel.txt'

                filepath1 = os.path.join('C:/Users/user/Desktop/semantic segmentation/testing image/UNet', filename)
                filepath2 = os.path.join('C:/Users/user/Desktop/semantic segmentation/testing image/UNet', pixel)
                with open(filepath1, "a+") as tmp:
                    tmp.write(f' i : {i}, mIoU : {mean_iou}\n')
                with open(filepath2, "a+") as tmp2:
                    tmp2.write(f' i : {i}, mIoU : {pixel_accuracy}')
                mean_iou = 0
                pixel_accuracy = 0
                cv2.imwrite(f'{checkpoint}/testing_image_{original_image_name}.png', img)
                #     if len(total_miou) == 1449:
                #
                #         iray = []
                #         count = 21
                #
                #         for m in range(21):
                #             iou = 0
                #             union = sum(total_miou[m, :]) + sum(total_miou[:, m]) - total_miou[m, m]
                #             intersection = total_miou[m, m]
                #             if union == 0:
                #                 iou = 0
                #             elif func[:, m].any() != 0:
                #                 iou = intersection / union
                #
                #             iray.append(iou)
                #
                #         iray = np.array(iray)
                #         mean_iou = sum(iray) / count
                #         print("mIoU : {}".format(mean_iou))
                #         total_miou = np.zeros((21, 21), dtype=int)
                #
                #     original_image_name = val_name[idx]
                #     cv2.imwrite(f'{checkpoint}/testing_image_{original_image_name}.png', img)
                #     # print("name : {} mIoU : {}".format(original_image_name, total_miou))
                #
                #     # with open('C:/Users/user/Desktop/semantic segmentation/testing image/psp50000_mIoU.txt',"a+") as tmp:
                #     #     tmp.write(f' i : {i}, name : {original_image_name}, loss : {loss.item()}, mIoU : {func}\n')
                #     if mean_iou is not None:
                #         if i <= 50000:
                #             filename = 'psp50000_mIoU.txt'
                #         elif i <= 100000:
                #             filename = 'psp100000_mIoU.txt'
                #         elif i <= 150000:
                #             filename = 'psp150000_mIoU.txt'
                #         else:
                #             filename = 'psp200000_mIoU.txt'
                #
                #         filepath = os.path.join('C:/Users/user/Desktop/semantic segmentation/testing image/PSPNet', filename)
                #         with open(filepath, "a+") as tmp:
                #             tmp.write(f' i : {i}, name, mIoU : {mean_iou}\n')
                #             # tmp.write(f' i : {i}, name : {original_image_name}, loss : {total_loss.item()}, mIoU : {mean_iou}\n')
                #
                # print("mIoU : {}".format(mean_iou))



















# path1 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation"
# path2 = "C:/Users/user/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
#
# print('Datasets loading...')
#
# train_txt = loader_train(path1, "train")
# val_txt = loader_test(path1, "train")
# new_train_name, new_train_image = loader2_train(path2, train_txt, "JPEGImages")
# new_val_name, new_val_image = loader2_val(path2, val_txt, "JPEGImages")
#
# train_gt_name, train_gt_image = loader_gt_train(path2, new_train_name, "SegmentationClass")
# val_gt_name, val_gt_image = loader_gt_val(path2, new_val_name,  "SegmentationClass")
#
#
#
# # using_train_gt_image = comp(train_gt_name, train_gt_image)
# # using_val_gt_image = comp(val_gt_name, val_gt_image)
# #
# # np.save("./256_train_gt_image.npy", using_train_gt_image)
# # np.save("./256_val_gt_image.npy", using_val_gt_image)
#
# # using_train_gt_image = np.load("train_gt_image.npy")
# # using_val_gt_image = np.load("val_gt_image.npy")
#
# using_train_gt_image = np.load("./256_train_gt_image.npy")
# using_val_gt_image = np.load("./256_val_gt_image.npy")
# print('Datasets loading complete')
#
#
# total_miou = []
# # test_tmp = using_val_gt_image[1]
# # model = UNET().to(device)
# model = UNet().to(device)
#
# # 저장된 모델 가중치 불러오기
# model.load_state_dict(torch.load('unet_model.pt'))
#
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.99, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.01, eps=1.0)
#
#
# iteration = 200000
# # bar = tqdm(range(iteration+1))
#
# with tqdm(range(45001, iteration + 1), dynamic_ncols=True, miniters=1) as pbar:
#     for i in pbar:
#         model.train()
#         new_train_image_tensor, new_train_gt_image = minibatch(new_train_image, using_train_gt_image, batch_size=32)
#         new_train_image_tensor = torch.from_numpy(new_train_image_tensor.astype(np.float32)).to(device)
#
#         new_train_gt_image = torch.from_numpy(new_train_gt_image.astype(np.float32)).to(device)
#         new_train_gt_image = new_train_gt_image.permute(0, 3, 1, 2)
#         new_train_image_tensor = new_train_image_tensor.permute(0, 3, 1, 2)
#
#         output = model(new_train_image_tensor).to(device)
#         # output = output.cpu().detach().numpy()
#         # loss = criterion(output, new_train_gt_image)
#         loss = F.cross_entropy(output, new_train_gt_image)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         if i > 50000:
#             optimizer.param_groups[0]['lr'] = 0.005
#         elif i > 100000:
#             optimizer.param_groups[0]['lr'] = 0.0005
#         elif i > 150000:
#             optimizer.param_groups[0]['lr'] = 0.00005
#         elif i > 200000:
#             optimizer.param_groups[0]['lr'] = 0.000005
#
#         if i % 100 == 0:
#             pbar.set_postfix({"Step": f"{i}/{iteration}", "Loss": f"{loss.item():.4f}"})
#
#         if i % 3000 == 0:
#             torch.save(model.state_dict(), 'unet_model_{i}.pt'.format(i=i+1))
#         if i % 3000 == 0 and i != 0:
#         # if i == 0:
#             model.eval()
#             correct = 0
#
#             with torch.no_grad():
#                 # temp = new_val_image[485]
#                 # temp_tensor = torch.from_numpy(temp.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
#                 # imag_tensor = temp_tensor / 255.0 * 2.0 - 1.0
#                 # temp_pred = model(imag_tensor)
#                 # temp_pred = temp_pred.cpu().numpy()
#                 #
#                 # temp_pred = temp_pred.transpose(0, 2, 3, 1)
#                 # temp_pred = temp_pred.squeeze(0)
#                 #
#                 # temp_img = seg_to_rgb(temp_pred)
#                 #
#                 # temp_pred = np.array(temp_pred)
#                 # temp_gt = np.array(using_val_gt_image[485])
#                 # func = mIoU(temp_pred, temp_gt)
#
#                 for idx in range(len(new_val_image)):
#                     imag = new_val_image[idx]
#                     height, width = imag.shape[0], imag.shape[1]
#                     # cv2.imshow("Image", imag)
#
#                     # cv2.imshow("Output", using_val_gt_image[idx])
#                     # cv2.waitKey(-1)
#
#                     imag_tensor = torch.from_numpy(imag.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
#                     imag_tensor = (imag_tensor / 255.0) * 2.0 - 1.0
#                     pred = model(imag_tensor) # 1, num_classes, height, width
#
#                     pred = pred.cpu().numpy()
#
#                     pred = pred.transpose(0, 2, 3, 1)
#                     pred = pred.squeeze(0)
#
#                     img = seg_to_rgb(img)
#                     # cv2.imshow("output", img)
#                     # cv2.waitKey(-1)
#                     # if i % 5000 == 0:
#                     #     cv2.imshow('output', img)
#                     #     cv2.waitKey(-1)
#
#                     checkpoint = f'C:/Users/user/Desktop/semantic segmentation/testing image/{i}'
#                     os.makedirs(checkpoint, exist_ok=True)
#
#                     pred = np.array(pred)
#                     gt = np.array(using_val_gt_image[idx])
#                     func = mIoU(pred, gt)
#
#                     original_image_name = new_val_name[idx]
#                     cv2.imwrite(f'{checkpoint}/testing_image_{original_image_name}_{func}.png', img)
#
#                     # pred = np.array(pred)
#                     # gt = np.array(using_val_gt_image[idx])
#                     # func = mIoU(pred, gt)
#                     print("name : {} mIoU : {}".format(original_image_name, func))
#                     # with open('C:/Users/user/Desktop/semantic segmentation/mIoU/mIoU.txt', "a+") as tmp:
#                     #     tmp.write(f' i : {i}, loss : {loss.item()}, mIoU : {func}\n')
#
#                     # if i <= 50000:
#                     #     with open('C:/Users/user/Desktop/semantic segmentation/mIoU/50000_mIoU.txt', "a+") as tmp:
#                     #         tmp.write(f' i : {i}, name : {original_image_name}, mIoU : {func}\n')
#
#                     if i <= 50000:
#                         with open('C:/Users/user/Desktop/semantic segmentation/mIoU/50000_mIoU.txt', "a+") as tmp:
#                             tmp.write(f' i : {i}, name : {original_image_name}, loss : {loss.item()}, mIoU : {func}\n')
#                     elif 50000 < i and i  <= 100000:
#                         with open('C:/Users/user/Desktop/semantic segmentation/mIoU/100000_mIoU.txt', "a+") as tmp:
#                             tmp.write(f' i : {i}, name : {original_image_name}, loss : {loss.item()}, mIoU : {func}\n')
#                     elif 100000 < i and i <= 150000:
#                         with open('C:/Users/user/Desktop/semantic segmentation/mIoU/150000_mIoU.txt', "a+") as tmp:
#                             tmp.write(f' i : {i}, name : {original_image_name}, loss : {loss.item()}, mIoU : {func}\n')
#                     elif 150000 < i and i <= 200000:
#                         with open('C:/Users/user/Desktop/semantic segmentation/mIoU/200000_mIoU.txt', "a+") as tmp:
#                             tmp.write(f' i : {i}, name : {original_image_name}, loss : {loss.item()}, mIoU : {func}\n')
#                     total_miou.append(func)
#                 # avg_miou = sum(total_miou) / len(total_miou)
#                 # with open('/home/aivs/PycharmProjects/GihyeonSeg/kh/total_mIoU.txt', "a+") as t_miou:
#                 #     t_miou.write(f' total_mIoU : {avg_miou}\n')
