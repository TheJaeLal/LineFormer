import cv2
import numpy as np

# Constants for line drawing and detection
LINE_THICKNESS = 1
LINE_SHIFT = 2
MIN_DISTANCE = 10 # Min distance between predicted lines
MIN_LINE_LENGTH = 5
SEARCH_THICKNESS = 10  # Thickness to search for white pixels

# Color constants
COLOR_HORIZONTAL = (255, 0, 0)  # Color for horizontal lines
COLOR_VERTICAL = (0, 255, 0)    # Color for vertical lines
COLOR_INTERSECTION = (0, 0, 255) # Color for intersections
COLOR_YELLOW = (0, 255, 255)     # Color for the vertical linepyt


INPUT_IMAGE = 'demo/image.png'
OUTPUT_IMAGE = 'demo/output_image_with_intersections.png'


def read_images(image_path):
    """Read the original and grayscale versions of the image."""
    image = cv2.imread(image_path)  # Original image
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale version
    return image, gray_image


def get_binary_map(gray_image):
    """Threshold the grayscale image to obtain a binary map."""
    _, binary_map = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary_map


def draw_horizontal_lines(image, binary_map, line_thickness, line_shift, min_distance, min_line_length, color):
    """Draw horizontal lines on the image based on the binary map."""
    height, width = binary_map.shape
    horizontal_lines = []  # Stores y-coordinates of horizontal lines
    last_y = -min_distance  # Initialize to a value smaller than any possible y-coordinate

    for y in range(height):
        count = 0  # Count consecutive white pixels
        for x in range(width):
            if binary_map[y, x] == 255:  # White pixel
                count += 1
            else:
                if count > min_line_length and y - last_y >= min_distance:
                    if y + line_shift < height:  # Check bounds
                        cv2.line(image, (0, y + line_shift), (width - 1, y + line_shift), color, line_thickness)
                        horizontal_lines.append(y + line_shift)
                        last_y = y
                count = 0  # Reset count if the sequence is broken

        # Check if the line reaches the end of the row
        if count > min_line_length and y - last_y >= min_distance:
            if y + line_shift < height:
                cv2.line(image, (0, y + line_shift), (width - 1, y + line_shift), color, line_thickness)
                horizontal_lines.append(y + line_shift)
                last_y = y

    return horizontal_lines


def draw_vertical_lines(image, binary_map, line_thickness, line_shift, min_distance, min_line_length, color):
    """Draw vertical lines on the image based on the binary map."""
    height, width = binary_map.shape
    vertical_lines = []  # Stores x-coordinates of vertical lines
    last_x = -min_distance  # Initialize to a value smaller than any possible x-coordinate

    for x in range(width):
        count = 0  # Count consecutive white pixels
        for y in range(height):
            if binary_map[y, x] == 255:  # White pixel
                count += 1
            else:
                if count > min_line_length and x - last_x >= min_distance:
                    if x + line_shift < width:  # Check bounds
                        cv2.line(image, (x + line_shift, 0), (x + line_shift, height - 1), color, line_thickness)
                        vertical_lines.append(x + line_shift)
                        last_x = x
                count = 0  # Reset count if the sequence is broken

        # Check if the line reaches the end of the column
        if count > min_line_length and x - last_x >= min_distance:
            if x + line_shift < width:
                cv2.line(image, (x + line_shift, 0), (x + line_shift, height - 1), color, line_thickness)
                vertical_lines.append(x + line_shift)
                last_x = x

    return vertical_lines


def mark_intersections(image, horizontal_lines, vertical_lines, color):
    """Mark intersections of horizontal and vertical lines with dots."""
    intersection_coordinates = []
    for i in range(min(len(horizontal_lines), len(vertical_lines))):
        intersection_point = (vertical_lines[i], horizontal_lines[i])  # (x, y) coordinate of the intersection
        intersection_coordinates.append(intersection_point)

        cv2.circle(image, intersection_point, radius=5, color=color, thickness=-1)
        print(f'Intersection {i + 1}: (X: {intersection_point[0]}, Y: {intersection_point[1]})')


    return intersection_coordinates


def draw_vertical_line_at_leftmost_white(image, binary_map, blue_line_y):
    """Draw a vertical line at the furthest left white pixel within a thickness of 10 pixels."""
    height, width = binary_map.shape
    leftmost_x = None

    # Search for the leftmost white pixel within SEARCH_THICKNESS above and below the blue line
    for y in range(max(0, blue_line_y - SEARCH_THICKNESS), min(height, blue_line_y + SEARCH_THICKNESS)):
        for x in range(width):
            if binary_map[y, x] == 255:  # Found a white pixel
                if leftmost_x is None or x < leftmost_x:
                    leftmost_x = x
                break  # Stop searching this row after finding the first white pixel

    # Draw a vertical line if a leftmost white pixel was found
    if leftmost_x is not None:
        cv2.line(image, (leftmost_x, 0), (leftmost_x, height - 1), COLOR_YELLOW, LINE_THICKNESS)
        print(f'Drawing vertical line at X: {leftmost_x}')
        return (leftmost_x, blue_line_y)


def draw_horizontal_line_at_bottom_most_white(image, binary_map, green_line_x):
    """Draw a horizontal line at the furthest bottom white pixel within a thickness of 10 pixels."""
    height, width = binary_map.shape
    bottom_most_y = None

    # Search for the bottom-most white pixel within SEARCH_THICKNESS above and below the green line
    for x in range(max(0, green_line_x - SEARCH_THICKNESS), min(height, green_line_x + SEARCH_THICKNESS)):
        # for y in range(height):
        for y in range(height - 1, -1, -1):
            if binary_map[y, x] == 255:  # Found a white pixel
                if bottom_most_y is None or y > bottom_most_y:
                    bottom_most_y = y
                break  # Stop searching this row after finding the first white pixel

    # Draw a horizontal line if a bottom-most white pixel was found
    if bottom_most_y is not None:
        cv2.line(image, (0, bottom_most_y), (width - 1, bottom_most_y), COLOR_YELLOW, LINE_THICKNESS)
        print(f'Drawing horizontal line at Y: {bottom_most_y}')
        return (green_line_x, bottom_most_y)




image, gray_image = read_images(INPUT_IMAGE)
binary_map = get_binary_map(gray_image)

horizontal_lines = draw_horizontal_lines(image, binary_map, LINE_THICKNESS, LINE_SHIFT, MIN_DISTANCE, MIN_LINE_LENGTH, COLOR_HORIZONTAL)
vertical_lines = draw_vertical_lines(image, binary_map, LINE_THICKNESS, LINE_SHIFT, MIN_DISTANCE, MIN_LINE_LENGTH, COLOR_VERTICAL)


intersection_coordinates = mark_intersections(image, horizontal_lines, vertical_lines, COLOR_INTERSECTION)

# Draw vertical line at the leftmost white pixel of the first blue line
if horizontal_lines:
    first_blue_line_y = horizontal_lines[0]  # Get the first horizontal blue line
    start = draw_vertical_line_at_leftmost_white(image, binary_map, first_blue_line_y)

if vertical_lines:
    last_green_line_x = vertical_lines[-1]  # Get the last vertical green line
    end = draw_horizontal_line_at_bottom_most_white(image, binary_map, last_green_line_x)

# intersection_coordinates.insert(0, start)
intersection_coordinates.append(end)

print(intersection_coordinates)

# Save or display the result
cv2.imwrite(OUTPUT_IMAGE, image)

