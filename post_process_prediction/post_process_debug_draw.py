import cv2

def draw_lines(img, group_dict, color=(0,255,0)):
    for item in group_dict:
        x, y = item['start_pos']
        x = int(x)
        y = int(y)

        x_end, y_end = item['end_pos']
        x_end = int(x_end)
        y_end = int(y_end)

        img = cv2.line(img, (x, y), (x_end, y_end), color, 1)
    return img


def draw_ponints(img, pnt_list, radius=2, color=(0, 255, 255)):
    for item in pnt_list:
        x, y = item
        img = cv2.circle(img, (x, y), radius=radius, color=color, thickness=-1)
    return img



def draw_km_lines(img, intersection_points, color=(0, 0, 255)):
    intersection_points = interpolate_points(intersection_points)

    prev_point = None
    for point in intersection_points:
        if prev_point is None:
            prev_point = point
            continue
        
        
        img = cv2.line(img, prev_point, point, color, 1)
        prev_point = point

    return img


def interpolate_points(points):
    interpolated = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        interpolated.append((x1, y1))
        interpolated.append((x1, y2))
    interpolated.append(points[-1])
    return interpolated