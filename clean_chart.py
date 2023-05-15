from PIL import Image, ImageDraw, ImageStat
import numpy as np
import cv2

def polygon2bbox(polygon_dic):
    x_coords = []
    y_coords = []
    for item in polygon_dic.keys():
        if "x" in item:
            x_coords.append(polygon_dic[item])
        elif "y" in item:
            y_coords.append(polygon_dic[item])

    x0 = min(x_coords)
    y0 = min(y_coords)

    x1 = max(x_coords)
    y1 = max(y_coords)
    return (x0, y0, x1, y1)


def get_legend_boxes(annot):
    # ============================================================
    # This is used to extract the text bbox and text role from task3 field in gt
    text_role_dic_id = {}
    id_text_bb_dic = {}

    #Used role is what role you would like to extract, 
    #if you only want to process the legend area, then legend_label & legend_title is enough
    for role in ['legend_label', 'legend_title']:  
        text_role_dic_id[role] = []


    text_block_list = annot["task3"]["input"]["task2_output"]["text_blocks"]
    for item in text_block_list:
        item_id = item["id"]
        if "polygon" in item:
            polygon_dic = item["polygon"]
            # Convert the polygon to the bbox
            (bbo_x0, bbox_y0, bbo_x1, bbo_y1) = polygon2bbox(polygon_dic)
            # id_text_bb_dic[item_id] = [x0, y0, x1, y1]

            poly_x0,poly_x1,poly_x2,poly_x3,poly_y0,poly_y1,poly_y2,poly_y3 = polygon_dic.values()
            id_text_bb_dic[item_id] = {"bbox":[bbo_x0, bbox_y0, bbo_x1, bbo_y1], "polygon":[poly_x0,poly_x1,poly_x2,poly_x3,poly_y0,poly_y1,poly_y2,poly_y3]}
        else:
            # Handle cleaning of adobe synth data
            id_text_bb_dic[item_id] = {"bbox":[item['bb']['x0'], item['bb']['y0'],
             item['bb']['x0']+item['bb']['width']-1, item['bb']['y0']+item['bb']['height']-1]}

    text_role_list = annot["task3"]["output"]["text_roles"]
    for item in text_role_list:
        role = item["role"]
        item_id = item["id"]
        # if role in used_role:
        if role not in text_role_dic_id.keys():
            text_role_dic_id[role] = []
        text_role_dic_id[role].append(item_id) 

    # ============================================================

    # ============================================================
    # Handle the legend patch and generate the legend area bbox


    legend_area_bb_list = []
    legend_patch_list = []

    for item in annot["task5"]["output"]["legend_pairs"]:
        x0 = item["bb"]["x0"]
        y0 = item["bb"]["y0"]
        x1 = x0 + item["bb"]["width"]
        y1 = y0 + item["bb"]["height"]

        legend_patch_list.append({"bbox":[x0,y0,x1,y1]})


    for legend_id in text_role_dic_id["legend_title"] + text_role_dic_id["legend_label"]:
        legend_area_bb_list.append(id_text_bb_dic[legend_id])
    for bbox_item in legend_patch_list:
        legend_area_bb_list.append(bbox_item)

    return legend_area_bb_list


def get_plot_area(annot):
    plot_area = annot['task6']['input']['task4_output']['_plot_bb']
    img_plot_area = {key: max(value, 0) for key, value in plot_area.items()}
    return img_plot_area


def get_legend_area(bbox_item_list):
    if bbox_item_list == [] or {}:
        return ()
    bbox_list = []
    # print(bbox_item_list)
    for bbox_item in bbox_item_list:
        # print(bbox_item["bbox"])
        bbox_list.append(bbox_item["bbox"])
    x0 = sorted(bbox_list, key = lambda i:i[0])[0][0]
    y0 = sorted(bbox_list, key = lambda i:i[1])[0][1]
    x1 = sorted(bbox_list, key = lambda i:i[2])[-1][2]
    y1 = sorted(bbox_list, key = lambda i:i[3])[-1][3]
    # color = (random.random()*255,random.random()*255,random.random()*255)
    # cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),color,2)
    # cv2.putText(img, bbox_name, (int(x0),int(y0)), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 1, cv2.LINE_AA)
    return (x0,y0,x1,y1)


def crop_to_plot_area(img, annot, crop_margin=1):
    plot_area = get_plot_area(annot)
    # print('crop_margin:', crop_margin)
    plot_x = plot_area['x0'] + crop_margin; plot_y = plot_area['y0'] + crop_margin;
    plot_w = plot_area['width'] - 2*crop_margin; plot_h = plot_area['height'] - 2*crop_margin
    # print(img.shape)
    # print(plot_area)
    # print(plot_x, plot_y, plot_w, plot_h)
    # crop out so we only have the plot area
    cropped_img = img[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w].copy()
    # print(cropped_img.shape)
    return cropped_img, (plot_x, plot_y)


def clean_nonline_elements(img, annot, legend_margin=1):
    im = Image.fromarray(img)
    js_obj = annot
    imd = ImageDraw.Draw(im)
    if js_obj['task6']['input']['task4_output'] is not None:
        plot_bb = js_obj['task6']['input']['task4_output']['_plot_bb']
        ploth, plotw , x0, y0 = plot_bb['height'], plot_bb['width'], plot_bb['x0'], plot_bb['y0']
        ctp = js_obj['task6']['input']['task1_output']['chart_type'] 
        tb =  js_obj['task6']['input']['task2_output']['text_blocks']
        legend_boxes = get_legend_boxes(annot=js_obj)
        # lp =  js_obj['task6']['input']['task5_output']['legend_pairs']
        ln_data = js_obj['task6']['output']['visual elements']['lines']
        x_axis = js_obj['task6']['input']['task4_output']['axes']['x-axis']
        y_axis = js_obj['task6']['input']['task4_output']['axes']['y-axis']
        for pt in x_axis :
            x_, y_ = pt['tick_pt']['x'], pt['tick_pt']['y']
            shape = [(x_, y_), (x_+2, y_+5)]
            # print('axis tb', [(x_, y_), (x_+2, y_+5)])
            cl = ImageStat.Stat(im).median
            imd.rectangle(shape, fill =tuple(cl),outline=None)
        for pt in y_axis :
            x_, y_ = pt['tick_pt']['x'], pt['tick_pt']['y']
            shape = [(x_, y_), (x_+5, y_+2)]
            # print('axis tb', [(x_, y_), (x_+2, y_+5)])
            cl = ImageStat.Stat(im).median
            imd.rectangle(shape, fill =tuple(cl),outline=None)

        ## remove text box
        for bx in tb :
            poly = bx['polygon'] if 'polygon' in bx else bx['bb']
            # print(poly)
            # Handle adobe synth format..
            if 'height' in poly:
                x_min = poly['x0']
                x_max = poly['x0'] + poly['width'] - 1
                y_min = poly['y0']
                y_max = poly['y0'] + poly['height'] - 1
            else:
                x_min = min(int(poly['x0']), int(poly['x1']), int(poly['x2']), int(poly['x3']))
                x_max = max(int(poly['x0']), int(poly['x1']), int(poly['x2']), int(poly['x3']))
                y_min = min(int(poly['y0']), int(poly['y1']), int(poly['y2']), int(poly['y3']))
                y_max = max(int(poly['y0']), int(poly['y1']), int(poly['y2']), int(poly['y3']))
            # print(x_min,x_max, y_min,y_max)
            # img_[y_min:y_max, x_min :x_max, :] = 255
            shape = [(x_min, y_min), (x_max, y_max)]
            # print('removed tb', [(x_min, y_min), (x_max, y_max)])
            cl = ImageStat.Stat(im).median
            imd.rectangle(shape, fill =tuple(cl),outline=None)
        
        ## remove legend
        if legend_boxes:
            x_min, y_min, x_max, y_max = get_legend_area(legend_boxes)
            x_min -= legend_margin; y_min -= legend_margin
            x_max += legend_margin; y_max += legend_margin

            shape = [(x_min, y_min), (x_max, y_max)]
            # print('removed leg', [(x_min, y_min), (x_max, y_max)])
            cl = ImageStat.Stat(im).median
            imd.rectangle(shape, fill = tuple(cl),outline=None)

        # imd.rectangle(shape, fill=(0,0,0),outline=None)
        # print('crop', (x0, y0, x0+plotw, y0+ploth))
        # im = im.crop((x0, y0, x0+plotw, y0+ploth))
        return np.array(im)
    
    return img

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # (Borrowed from imutils)initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def _get_interpolation(inter_string):
    
    # In case it's already an interpolation code..
    if isinstance(inter_string, int):
        return inter_string
    
    inter_string = inter_string.lower().strip()
    inter_methods = {'area':cv2.INTER_AREA,
                     'linear':cv2.INTER_LINEAR,
                     'cubic':cv2.INTER_CUBIC,
                     'nearest':cv2.INTER_NEAREST}
    
    if inter_string not in inter_methods:
        raise Exception("Unknown Interpolation Method: '{}'".format(inter_string))
    
    return inter_methods[inter_string]


def padd_square(img, desired_size, padd_color=255):
    """
        resize and square padd
        img: np.array of image shaped (h,w,c)
        desired_size: int size of the image after resize and padding
    """
    
    if padd_color==255 and img.ndim == 3:
        padd_color = [255, 255, 255]
    
    size = img.shape[:2]
    delta_w = desired_size - size[1]
    delta_h = desired_size - size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=padd_color)

    return new_img, (left, top)


def get_clean_input(img, annot, crop_to_plot=True, remove_text_legend=True, legend_margin=1, crop_margin=1, max_size=512, padd=True):
    """
        img: rgb image of line chart
        annot: json obj of PMC groundtruth
        max_size: resize max dimension of input to this size (maintaining aspect ratio)
        padd: whether to square padd the image after resizing
        crop_to_plot: whether to crop the chart image to plot area based on annotation provided
        remove_text_legend: whether to remove the text and legend boxes from chart image based on annotation provided
        returns: rgb image of cleaned line chart with plot area cropped
    """

    clean_img = img if not remove_text_legend else clean_nonline_elements(img, annot, legend_margin)
    
    if crop_to_plot:
        clean_img, (tx_crop, ty_crop)= crop_to_plot_area(clean_img, annot, crop_margin)
    else:
        tx_crop, ty_crop = 0,0
    
    sx, sy= 1,1
    tx_padd, ty_padd = 0,0

    h_cropped, w_cropped = clean_img.shape[:2]
    if max_size:
        if clean_img.shape[0] > clean_img.shape[1]:
            clean_img = resize(clean_img, height=max_size)
        else:
            clean_img = resize(clean_img, width=max_size)
        
        sx, sy = float(clean_img.shape[1])/w_cropped, float(clean_img.shape[0])/h_cropped

    if padd:
        clean_img, (tx_padd, ty_padd) = padd_square(clean_img, max_size)

    transformation = (sx, sy, tx_crop, ty_crop, tx_padd, ty_padd)
    
    return clean_img, transformation

# with open(annot_path, 'r') as f:
    # annot = json.load(f)