import mmcv
import itertools
import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from mmdet.apis import (inference_detector, init_detector)

# import sys
# sys.path.append("/home/csgrad/jayashok/Mask2Former/")
from clean_chart import get_clean_input
import line_utils

import copy


def hsv_to_bgr(h, s, v):
    # Get RGB values
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c

    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 1/3:
        r, g, b = x, c, 0
    elif h < 0.5:
        r, g, b = 0, c, x
    elif h < 2/3:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    # Scale RGB values to 0-255 range and convert to integers
    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return (b, g, r)


def get_distinct_colors(n): 
    huePartition = 1.0 / (n + 1) 
    return (hsv_to_bgr(huePartition * value, 1.0, 1.0) for value in range(0, n))


def load_model(config, ckpt, device):
    global model
    model = init_detector(config, ckpt, device=device)
    return 


def do_instance(model, img, score_thr=0.3):
    # test a single image
    result = inference_detector(model, img)
    return parse_result(result, score_thr)

def parse_result(result, score_thresh=0.3):
    line_data = result[0][0]
    # print(type(result))
    bbox, masks = line_data[0][0], line_data[1][0]
    inst_masks = list(itertools.compress(masks, ((bbox[:, 4] > score_thresh).tolist())))
    return inst_masks


def draw_lines(img, masks):
    annot_img = img.copy()
    colors = list(get_distinct_colors(len(masks)))
    color_map = dict(zip(range(len(masks)), colors))

    # show_img(img, is_bgr=True, title='original')
    for idx, mask in enumerate(masks):
        annot_img[mask] = color_map[idx]
        # show_img(img, is_bgr=True, title=f'line_{idx+1}')
    
    return annot_img

def connect_lines(img):
    #img = cv2.imread('line_join_test2.png', 0)     # grayscale image
    #img1 = cv2.imread('line_join_test2.png', 1)    # color image
    
    th = cv2.threshold(img.astype(np.uint8), 150, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)) #(19, 19)
    img = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel)
    
    cnts1 = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts1[0]     # storing contours in a variable


    for i in range(len(cnts)):
        min_dist = max(img.shape[0], img.shape[1])

        cl = []

        ci = cnts[i]
        ci_left = tuple(ci[ci[:, :, 0].argmin()][0])
        ci_right = tuple(ci[ci[:, :, 0].argmax()][0])
        ci_top = tuple(ci[ci[:, :, 1].argmin()][0])
        ci_bottom = tuple(ci[ci[:, :, 1].argmax()][0])
        ci_list = [ci_bottom, ci_left, ci_right, ci_top]

        for j in range(i + 1, len(cnts)):
            cj = cnts[j]
            cj_left = tuple(cj[cj[:, :, 0].argmin()][0])
            cj_right = tuple(cj[cj[:, :, 0].argmax()][0])
            cj_top = tuple(cj[cj[:, :, 1].argmin()][0])
            cj_bottom = tuple(cj[cj[:, :, 1].argmax()][0])
            cj_list = [cj_bottom, cj_left, cj_right, cj_top]

            for pt1 in ci_list:
                for pt2 in cj_list:
                    dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))    
                    if dist < min_dist:
                        min_dist = dist             
                        cl = []
                        cl.append([pt1, pt2, min_dist])
                        
        if len(cl) > 0:
            cv2.line(img, cl[0][0], cl[0][1], (255, 255, 255), thickness = 5)

    img = img//255
    img = skeletonize(img).astype(np.uint8)
    img = img * 255

    return img


def interpolate(line_ds, inter_type='linear'):
    """
    pred_ds: predicted data series
    inter_type: type of interpolation linear or cubic_spline
    returns list of interpolation objects for each line in the mask

    """

    x = []
    y = []

    for pt in line_ds:
        x.append(pt['x'])
        y.append(pt['y'])

    # Remove duplicates
    unique_x = []
    unique_y = []
    
    for i in range(len(x)):
        if x.count(x[i]) == 1:
            unique_x.append(int(x[i]))
            unique_y.append(int(y[i]))
    
    if len(unique_x) < 2:
        return line_ds
    
    # Interpolate    
    if inter_type == 'linear':
        inter = interp1d(unique_x, unique_y)
    if inter_type == 'cubic_spline':
        inter = CubicSpline(unique_x, unique_y)

    inter_line_ds = []
    x_min = min(unique_x)
    x_max = max(unique_x)

    for x in range(x_min, x_max+1):
        inter_line_ds.append({"x":x, "y":int(inter(x))})

    return inter_line_ds

 
def post_process(inst_masks):
    post_process_mask = []
    for i in range(len(inst_masks)):
        post_process_mask.append(connect_lines(inst_masks[i]))

    return post_process_mask

def rescale_pred_ds(ds, transformation):
    ds = copy.deepcopy(ds)
    (sx, sy, tx_crop, ty_crop, tx_padd, ty_padd) = transformation
    
    # print(sx, sy, tx_crop, ty_crop, tx_padd, ty_padd)
    for ln in ds:
        for pt in ln:
            pt['x'] = int((pt['x']-tx_padd) / sx) + tx_crop
            pt['y'] = int((pt['y']-ty_padd) / sy) + ty_crop
    return ds

def get_dataseries(img, annot=None, to_clean=False, post_proc=False, mask_kp_sample_interval=10, return_masks=False):
    """
        img: chart image as numpy array (3 channel) 
        annot: json annot object in PMC format (required for cleaning the chart image before data extraction)
        mask_kp_sample_interval: interval to sample points from predicted line mask to get data series
        returns data series in pmc task 6a format ('visual elements') => list of lines, each a list of {x:, y: } points w.r.t original image
    """
    global model
    # clean the image
    # save the transformation for clean image
    if to_clean:
        clean_img, transformation = get_clean_input(img, annot)
    else:
        clean_img = img
    # Image.fromarray(clean_img)

    # get inference masks
    inst_masks = do_instance(model, clean_img, score_thr=0.3)
    # return inst_masks
    mask_thresh = 0.5
    inst_masks = [(line_mask > mask_thresh).astype(np.uint8)*255 for line_mask in inst_masks]
    # for line_masks in inst_masks:
    #     line_masks[:, line_masks.sum(axis=1)>1] = 0
    # plt.imshow(inst_masks[0])
    # plt.show()
    # return inst_masks
    if post_proc:
        inst_masks = post_process(inst_masks)
    # annot_img = infer.draw_lines(clean_img, inst_masks)
    # plt.imshow(annot_img)

    # inference data series    
    pred_ds = []
    # print(len(inst_masks))
    for line_mask in inst_masks:
        # print(line_mask.shape)
        # print(len(line_mask))
        # print(line_mask)
        x_range = line_utils.get_xrange(line_mask)
        line_ds = line_utils.get_kp(line_mask, interval=mask_kp_sample_interval, x_range=x_range, get_num_lines=False, get_center=True)
        
        line_ds = interpolate(line_ds, inter_type='linear')

        pred_ds.append(line_ds)

    # Reverse that transformation on pred-ds
    if to_clean: 
        pred_ds = rescale_pred_ds(pred_ds, transformation)
    if return_masks:
        return pred_ds, inst_masks
    else:
        return pred_ds


# Swin Transformer Backbone
CONFIG = "lineformer_swin_t_config.py"
CKPT = "best_segm_mAP_iter_17000.pth"
DEVICE = 'cpu'
model = load_model(CONFIG, CKPT, DEVICE)
print('Loaded Model:', model)
# if __name__ == '__main__':
#     img_path = "/a2il/data/ChartAnalysis/pmc_2020_split4/val_images/PMC3169544___pgen.1002274.g005.png"
#     #Note: Image is Loaded as BGR to RGB
#     img = mmcv.imread(img_path)
#     inst_masks = do_instance(model, img, score_thr=0.3)
#     annot_img = draw_lines(img, inst_masks)
#     post_processed_mask = post_process(inst_masks)

#     for i in range(len(post_processed_mask)):
#         cv2.imwrite(str(i)+".jpg", post_processed_mask[i])
#         #cv2.imwrite(str(i)+".jpg", inst_masks[i].astype(np.uint8)*255)
        