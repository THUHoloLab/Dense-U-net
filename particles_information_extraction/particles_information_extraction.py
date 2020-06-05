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

    img = cv2.imread('D:/pycharm/up_down_code/Dense-U-net/predict_data/denseunet_%d.png' % (i + 1))
    gray_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, th1 = cv2.threshold(hsv_mask, 1, 255, cv2.THRESH_BINARY)  #Fixed threshold binarization
    gray_mask_new = cv2.cvtColor(th1, cv2.COLOR_BGR2GRAY)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Square
    erosion = cv2.erode(gray_mask_new, rect, iterations=1)  # Corrosion treatment
    dilation = cv2.dilate(erosion, rect, iterations=1)  # 膨胀处理
    binary_img = Two_Pass(dilation, NEIGHBOR_HOODS_8)
    #
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    depth_all = []
    r_all = []
    x_all = []
    y_all = []
    for c in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[c])
        R = min(w, h)
        if R > 5:
            r_mid = int(R / 2)
            x_mid = x + r_mid
            y_mid = y + r_mid
            x_int = int(x)
            x_end = int(x + R)
            y_int = int(y)
            y_end = int(y + R)
            depth_list = gray_mask[y_int:y_end, x_int:x_end]

            depth_list = gray_mask[y_int:y_end, x_int:x_end]
            depth_list = np.ravel(depth_list)
            depth_list = np.sort(depth_list)
            depth_list = depth_list[-(R + int(R / 2)):-int(R / 2)]
            d = round(np.mean(depth_list))

            depth_all.append(d)
            r_all.append(R)
            x_all.append(x_mid)
            y_all.append(y_mid)
            cv2.rectangle(img, (x, y), (x + R, y + R), (0, 255, 255), 1)

        # cv2.imshow('img',img)
        # cv2.imshow('hsv_mask', hsv_mask)
        # cv2.imshow('th1', th1)
        # cv2.imshow('dilation', dilation)
        # cv2.imshow('img', binary_img)
        # cv2.imshow('gray_mask',gray_mask)
        # cv2.waitKey(0)
    cv2.imwrite('./predict_img/%d.png' % (i + 1), img)
    np.savetxt('./predict_data/depth_%d.txt'%i, depth_all, fmt='%f')
    np.savetxt('./predict_data/x_%d.txt'%i, x_all, fmt='%f')
    np.savetxt('./predict_data/y_%d.txt'%i, y_all, fmt='%f')
    np.savetxt('./predict_data/r_%d.txt'%i, r_all, fmt='%f')
    print(i)



