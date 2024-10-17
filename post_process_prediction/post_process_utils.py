import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage import img_as_bool, morphology



def load_binary_mask(path, min_px_value=127):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, min_px_value, 255, cv2.THRESH_BINARY)
    return binary_mask


def get_skeleton(mask):
    # Convert to a binary image
    binary_mask = img_as_bool(mask)
    skeleton = morphology.skeletonize(binary_mask)
    skeleton = (skeleton * 255).astype(np.uint8)

    return skeleton


def find_starting_point(binary_mask):
    # Find most top left point
    y, x = np.where(binary_mask == 255)
    coordinates = sorted(zip(x, y), key=lambda coord: (coord[0], coord[1]))
    return coordinates[0]


def get_kp_first_hit_x(binary_mask, steps=10):
    height, width = binary_mask.shape
    starting_point = find_starting_point(binary_mask)

    kps = []

    for x in range(starting_point[0], width-1, steps):
        for y in range(0, height-1):
            if binary_mask[y, x] == 255:
                kps.append((x, y))
                break

    return kps


def get_kp_first_hit_y(binary_mask, steps=10):
    height, width = binary_mask.shape
    starting_point = find_starting_point(binary_mask)

    kps = []

    for y in range(starting_point[1], width, steps):
        for x in range(width-1, 0, -1):
            if binary_mask[y, x] == 255:
                kps.append((x, y))
                break

    return kps


def extend_lines_x(binary_mask, group_dict):
    new_list = []
    height, width = binary_mask.shape
    for item in group_dict:
        _item = item.copy()
        x, y = (int(coord) for coord in item['start_pos'])

        _item['start_pos'] = (x, y)
        _item['end_pos'] = (width-1, y)

        new_list.append(_item)

    return new_list


def extend_lines_y(binary_mask, group_dict):
    new_list = []
    height, width = binary_mask.shape
    for item in group_dict:
        _item = item.copy()
        x, y = (int(coord) for coord in item['end_pos'])

        _item['start_pos'] = (x, y)
        _item['end_pos'] = (x, 0)

        new_list.append(_item)

    return new_list



def find_intersections(horizontal_lines, vertical_lines):
    intersections = []
    
    # Loop through each horizontal line
    for h_line in horizontal_lines:
        h_y = h_line['start_pos'][1]  # y-coordinate of the horizontal line
        h_x_start = min(h_line['start_pos'][0], h_line['end_pos'][0])
        h_x_end = max(h_line['start_pos'][0], h_line['end_pos'][0])
        
        # Loop through each vertical line
        for v_line in vertical_lines:
            v_x = v_line['start_pos'][0]  # x-coordinate of the vertical line
            v_y_start = min(v_line['start_pos'][1], v_line['end_pos'][1])
            v_y_end = max(v_line['start_pos'][1], v_line['end_pos'][1])
            
            # Check if the lines intersect
            if h_x_start <= v_x <= h_x_end and v_y_start <= h_y <= v_y_end:
                # Intersection point is (v_x, h_y)
                intersections.append((v_x, h_y))
    
    return intersections



def keep_first_intersection_point_found_on_x_group(points, group_x_dict):
    points_to_keep = []

    for item in group_x_dict:
        line_x, line_y = item['start_pos']
        for pnt in points: 
            pnt_x, pnt_y = pnt
            if pnt_y == line_y:
                points_to_keep.append(pnt)
                break
    
    return points_to_keep
