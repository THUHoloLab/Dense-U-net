#coding:utf-8
import cv2
import numpy as np
from numpy.random import poisson as poisson


#Connected Component Analysis
# 4 neighborhood connected domain and 8 neighborhood connected domain
NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1, 0], [0, 0], [1, 0],
             [-1, 1], [0, 1], [1, 1]]
def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num

            points[index].append([row, col])

    return binary_img, points


def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows - 1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols - 1, -1, -1]

    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
                # break
            for offset in offsets:
                neighbor_row = min(max(0, row + offset[0]), rows - 1)
                neighbor_col = min(max(0, col + offset[1]), cols - 1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx

            binary_img[row][col] = label

    return binary_img

# binary_img: bg-0, object-255; int
def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)

    return binary_img


def recursive_seed(binary_img: np.array, seed_row, seed_col, offsets, num, max_num=100):
    rows, cols = binary_img.shape
    binary_img[seed_row][seed_col] = num
    for offset in offsets:
        neighbor_row = min(max(0, seed_row + offset[0]), rows - 1)
        neighbor_col = min(max(0, seed_col + offset[1]), cols - 1)
        var = binary_img[neighbor_row][neighbor_col]
        if var < max_num:
            continue
        binary_img = recursive_seed(binary_img, neighbor_row, neighbor_col, offsets, num, max_num)

    return binary_img

def Seed_Filling(binary_img, neighbor_hoods, max_num=100):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    num = 1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var <= max_num:
                continue
            binary_img = recursive_seed(binary_img, row, col, offsets, num, max_num=100)
            num += 1
    return binary_img

for i in range(2):

    img_label = cv2.imread('D:/pycharm/up_down_code/Layer-oriented-algorithm-and-ground-true/text_data/ground_true/label/label/%d.tif' % (i + 1))
    gray_mask_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
    gray_mask_label = np.array(gray_mask_label)
    hsv_mask_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2HSV)
    ret, th1_label = cv2.threshold(hsv_mask_label, 1, 255, cv2.THRESH_BINARY)  # 固定阈值二值化
    gray_mask_new_label = cv2.cvtColor(th1_label, cv2.COLOR_BGR2GRAY)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 方形
    erosion_label = cv2.erode(gray_mask_new_label, rect, iterations=1)  # 腐蚀处理
    dilation_label = cv2.dilate(erosion_label, rect, iterations=1)  # 膨胀处理
    # binary_img_label = Seed_Filling(dilation_label, NEIGHBOR_HOODS_8)
    binary_img_label = Two_Pass(dilation_label, NEIGHBOR_HOODS_8)
    contours_label, _ = cv2.findContours(binary_img_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    depth_all_label = []
    r_all_label = []
    x_all_label = []
    y_all_label = []
    for c in range(0, len(contours_label)):
        x, y, w, h = cv2.boundingRect(contours_label[c])
        R = max(w, h)
        r_mid = int(R / 2)
        x_mid = x + r_mid
        y_mid = y + r_mid
        x_int = int(x)
        x_end = int(x + R)
        y_int = int(y)
        y_end = int(y + R)
        depth_list = gray_mask_label[y_int:y_end, x_int:x_end]

        depth_list = gray_mask_label[y_int:y_end, x_int:x_end]
        depth_list = np.ravel(depth_list)
        depth_list = np.sort(depth_list)
        depth_list = depth_list[-(R + int(R / 2)):-int(R / 2)]
        d = round(np.mean(depth_list))

        depth_all_label.append(d)
        r_all_label.append(R)
        x_all_label.append(x_mid)
        y_all_label.append(y_mid)
        cv2.rectangle(img_label, (x, y), (x + R, y + R), (0, 0, 255), 1)

        # cv2.imshow('img',img)
        # cv2.imshow('hsv_mask', hsv_mask)
        # cv2.imshow('th1', th1)
        # cv2.imshow('dilation', dilation)
        # cv2.imshow('img', binary_img)
        # cv2.imshow('gray_mask',gray_mask)
        # cv2.waitKey(0)
    cv2.imwrite('./label_img/%d.png' % (i + 1), img_label)
    np.savetxt('./label_data/depth_%d.txt'%i, depth_all_label, fmt='%f')
    np.savetxt('./label_data/x_%d.txt'%i, x_all_label, fmt='%f')
    np.savetxt('./label_data/y_%d.txt'%i, y_all_label, fmt='%f')
    np.savetxt('./label_data/r_%d.txt'%i, r_all_label, fmt='%f')
    print(i)



