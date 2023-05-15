import numpy as np
import cv2
from  matplotlib import pyplot as plt
import scipy

# For line interpolation:
# Comment this out if not using 'get_interp_points' function
from bresenham import bresenham

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


def is_color(img):
    if img.ndim <=2:
        return False

    # img.ndim >=3
    if img.shape[-1] == 1:
        return False

    return True
    

def show_img(img, color='gray', is_bgr=False, title='', figsize=None, final_show=True):
    """Show image using plt."""
    

    if figsize:
        if not isinstance(figsize, (tuple, list)):
            figsize = (figsize, figsize)
        plt.figure(figsize=figsize)

    if is_color(img) and is_bgr:
        img = img[...,::-1]

    params = {'cmap':color}

    if img.dtype == np.uint8:
        params.update({'vmin': 0, 'vmax': 255})

    plt.imshow(img, **params)
    plt.title(title)
    if final_show:
        plt.show()
    
    return


def draw_xrange(img, xrange):
    annot_img = img.copy()
    im_h, im_w = annot_img.shape[:2]
    annot_img = cv2.line(annot_img, (xrange[0], 0), (xrange[0], im_h), (0,0,255), thickness=1)
    annot_img = cv2.line(annot_img, (xrange[1], 0), (xrange[1], im_h), (0,0,255), thickness=1)
    return annot_img

def get_xrange(bin_line_mask):
    """
        bin_line_mask: np.ndarray => black and white binary mask of line
        black => background => 0
        white => foregrond line pixel => 255
        returns: (x_start, x_end) where x_start and x_end represent the starting and ending points 
                for the binary line segment
    """
    # print(bin_line_mask.sum(axis=0))
    # np.save("problem_mask.npy", bin_line_mask)
    smooth_signal = scipy.signal.medfilt(bin_line_mask.sum(axis=0), kernel_size=5)
    # print(smooth_signal.shape)
    # print(smooth_signal)
    # print(np.nonzero(smooth_signal))
    x_range = np.nonzero(smooth_signal)
    if len(x_range) and len(x_range[0]): # To handle cases with empty masks
        x_range = x_range[0][[0, -1]]
    else:
        x_range = None
    return x_range

def get_kp(line_img, interval=10, x_range=None, get_num_lines=False, get_center=True):
    """
        line_img: np.ndarray => black and white binary mask of line
        black => background => 0
        white => foregrond line pixel => 255
        interval: delta_x at which x,y points are sampled across the line_img
        x_range: Range of x values, [xmin, xmax), within which pred points (x,y) are to be sampled
        returns: a list [{'x': <x_val>, 'y': <y_val>}, ....] of line points found in the binary line_img
    """

    im_h, im_w = line_img.shape[:2]
    kps = []
    # delta = 2
    if x_range is None:
        x_range = (0, im_w)
    
    # track the number of vertical binary components found at every x => estimate num lines
    num_comps = [] 
    for x in range(x_range[0], x_range[1], interval):
        # get the corresponding white pixel in this column
        fg_y = []
        fg_y_center = []
        all_y_points = np.where(line_img[:, x] == 255)[0]
        if all_y_points.size != 0:
            fg_y.append(all_y_points[0])
            y = all_y_points[0]
            n_comps = 1
            for idx in range(1, len(all_y_points)):
                y_next = all_y_points[idx]
                # print(y, y_next)
                if abs(y_next - y) > 2:
                    n_comps += 1
                    # break found b/w y_next and y, separate components
                    if fg_y[-1] != y:
                        # handle the case where (first component itself is broken, i.e found break at idx=1)
                        fg_y_center.append(round(y + fg_y[-1])//2)
                        fg_y.append(y)
                    else:
                        fg_y_center.append(y)
                        
                    fg_y.append(y_next)
                    
                y = y_next
                # print(fg_y,'\n', fg_y_center,  '\n')
              
            # print('last_point', y, y_next)
            if fg_y[-1] != y:
                # add the last point
                fg_y_center.append(round(y + fg_y[-1])//2)
                fg_y.append(y)
            else:
                fg_y_center.append(y)
                    
            num_comps.append(n_comps)
        
        if (fg_y or fg_y_center) and (n_comps==1):
            if get_center:
                kps.extend([{'x':float(x), 'y':y} for y in fg_y_center])
            else:
                kps.extend([{'x':float(x), 'y':y} for y in fg_y])
        
    res = kps
    
    if get_num_lines:
        res = kps, int(np.percentile(num_comps, 85))
        
    return res


def draw_edge(img, edge):
    inter_points = get_interp_points(edge[0], edge[1])
    # print(inter_points)
    # print(len(inter_points), inter_points)
    annot_img = draw_kps(img, array_to_points(inter_points), color=(255,0,0))
    return annot_img

def draw_kps(img, kps, color=(0,255,0), classes=None, **draw_options):
    if is_color(img):
        annot_img = img.copy()
    else:
        annot_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if classes is None:
        classes = [0]*len(kps)
        color_map = {0: color}
    else:
        colors = list(get_distinct_colors(classes.max()+1))
        color_map = dict(zip(range(classes.max()+1), colors))
        # print(color_map)
    for idx, kp in enumerate(kps):
        options = dict(color=color_map[classes[idx]], markerType=cv2.MARKER_CROSS, markerSize=2, thickness=2, line_type=8)
        options.update(draw_options)
        annot_img = cv2.drawMarker(annot_img, (int(kp['x']), int(kp['y'])), **options)
    return annot_img

def points_to_array(pred_ds):
    res = []
    for line in pred_ds:
        line_arr = []
        for pt in line:
            line_arr.append([pt['x'], pt['y']])
        res.append(line_arr)
    return res

# res = line_utils.draw_lines(img, points_to_array(pred_ds))
def draw_lines(img, lines, classes=None):
    if is_color(img):
        annot_img = img.copy()
    else:
        annot_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if classes is None:
        classes = list(range(len(lines)))
    if len(classes):
        colors = list(get_distinct_colors(max(classes)+1))
        color_map = dict(zip(range(max(classes)+1), colors))
        # print('color_map:', color_map)
        
        for line_idx, line in enumerate(lines):
            options = dict(color=color_map[classes[line_idx]], thickness=2)
            drawing_lines = []
            for pt_idx in range(len(line)-1):
                drawing_lines.append([line[pt_idx], line[pt_idx+1]])
            annot_img = cv2.polylines(annot_img, np.array(drawing_lines), isClosed=False, **options)

    return annot_img


# Get the line points that would lie between ptA and ptB, according to the bresenham algorithm
def get_interp_points(ptA, ptB, thickness=1):
    # x_interp = np.arange(ptA[0], ptB[0])
    # y_interp = np.interp(x_interp, [ptA[0], ptB[0]], [ptA[1], ptB[1]]).round().astype(int)
    
    points = []
    delta_range = (-thickness//2, thickness//2)

    for delta in range(delta_range[0], delta_range[1]+1):
        points.extend(list(bresenham(ptA[0], ptA[1]+delta, ptB[0], ptB[1]+delta)))

    inter_points = np.array(points)
    
    # inter_points = np.stack([x_interp,y_interp], axis=-1)
    return inter_points

def array_to_points(pts_arr):
    pts = [{'x': pt[0], 'y': pt[1]} for pt in pts_arr]
    return pts