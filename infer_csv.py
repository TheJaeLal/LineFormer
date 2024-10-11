import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

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
COLOR_START = (0, 255, 255) 
COLOR_END = (0, 255, 255) 


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


def get_horizontal_lines(image, binary_map, line_thickness, line_shift, min_distance, min_line_length, color):
    """Get horizontal lines on the image based on the binary map."""
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
                        horizontal_lines.append(y + line_shift)
                        last_y = y
                count = 0  # Reset count if the sequence is broken

        # Check if the line reaches the end of the row
        if count > min_line_length and y - last_y >= min_distance:
            if y + line_shift < height:
                horizontal_lines.append(y + line_shift)
                last_y = y

    return horizontal_lines


def get_vertical_lines(image, binary_map, line_thickness, line_shift, min_distance, min_line_length, color):
    """Get vertical lines on the image based on the binary map."""
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
                        vertical_lines.append(x + line_shift)
                        last_x = x
                count = 0  # Reset count if the sequence is broken

        # Check if the line reaches the end of the column
        if count > min_line_length and x - last_x >= min_distance:
            if x + line_shift < width:
                vertical_lines.append(x + line_shift)
                last_x = x

    return vertical_lines


def mark_intersections(image, horizontal_lines, vertical_lines, color):
    """Get horizontal and vertical lines"""
    intersection_coordinates = []
    for i in range(min(len(horizontal_lines), len(vertical_lines))):
        intersection_point = (vertical_lines[i], horizontal_lines[i])  # (x, y) coordinate of the intersection
        intersection_coordinates.append(intersection_point)

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
        # cv2.line(image, (leftmost_x, 0), (leftmost_x, height - 1), COLOR_YELLOW, LINE_THICKNESS)
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
        # cv2.line(image, (0, bottom_most_y), (width - 1, bottom_most_y), COLOR_YELLOW, LINE_THICKNESS)
        print(f'Drawing horizontal line at Y: {bottom_most_y}')
        return (green_line_x, bottom_most_y)


def draw_indicators(
    image,
    binary_mask,
    intersection_coordinates,
    draw_horizontal_lines = False,
    draw_vertical_lines = False,
    draw_intersection = False,
    draw_start_pnt = False,
    draw_end_pnt = False,
):  
    new_image = image.copy()
    height, width, _ = new_image.shape

    for idx, (x, y) in enumerate(intersection_coordinates):
        
        if draw_horizontal_lines and idx != len(intersection_coordinates) - 1:
            cv2.line(new_image, (0, y), (width - 1, y), COLOR_HORIZONTAL, LINE_THICKNESS)
        
        if draw_vertical_lines and idx != 0:
            cv2.line(new_image, (x, 0), (x, height - 1), COLOR_VERTICAL, LINE_THICKNESS)

        if draw_intersection:
            cv2.circle(new_image, (x, y), radius=5, color=COLOR_INTERSECTION, thickness=-1)

        if draw_start_pnt and idx == 0:
            cv2.circle(new_image, (x, y), radius=5, color=COLOR_START, thickness=-1)

        if draw_end_pnt and idx == len(intersection_coordinates) - 1:
            cv2.circle(new_image, (x, y), radius=5, color=COLOR_END, thickness=-1)
    
    return new_image


def get_km_df_from_intersections(intersection_coordinates):
    intersection_coordinates = intersection_coordinates[1:]
    # Extract times (X) and survival probabilities (Y)
    x_values = [coord[0] for coord in intersection_coordinates]
    y_values = [coord[1] for coord in intersection_coordinates]

    y_max = y_values[0]  # Corresponds to 100% survival
    y_min = y_values[-1]  # Corresponds to 0% survival
    survival_probabilities = [(y_max - y) / (y_max - y_min) for y in y_values]

    # Prepare the data for Kaplan-Meier Fitting
    event_times = np.array(x_values)  # Time of events
    observed_deaths = [1] * (len(event_times) - 1) + [0]  # 1 for deaths, 0 for censoring at the end

    # Create a DataFrame to structure the data for CSV
    df = pd.DataFrame({
        'time': event_times,
        'event_observed': observed_deaths,
        'survival_probability': survival_probabilities
    })

    return df


def plot_kaplan_meier(df):
    # Fit the Kaplan-Meier estimator
    kmf = KaplanMeierFitter()
    kmf.fit(df['time'], event_observed=df['event_observed'])

    # Plot the Kaplan-Meier curve
    kmf.plot(ci_show=False)
    plt.title("Kaplan-Meier Curve from Coordinates")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.show()



image, gray_image = read_images(INPUT_IMAGE)
binary_map = get_binary_map(gray_image)

horizontal_lines = get_horizontal_lines(image, binary_map, LINE_THICKNESS, LINE_SHIFT, MIN_DISTANCE, MIN_LINE_LENGTH, COLOR_HORIZONTAL)
vertical_lines = get_vertical_lines(image, binary_map, LINE_THICKNESS, LINE_SHIFT, MIN_DISTANCE, MIN_LINE_LENGTH, COLOR_VERTICAL)


intersection_coordinates = mark_intersections(image, horizontal_lines, vertical_lines, COLOR_INTERSECTION)

# Draw vertical line at the leftmost white pixel of the first blue line
if horizontal_lines:
    first_blue_line_y = horizontal_lines[0]  # Get the first horizontal blue line
    start = draw_vertical_line_at_leftmost_white(image, binary_map, first_blue_line_y)

if vertical_lines:
    last_green_line_x = vertical_lines[-1]  # Get the last vertical green line
    end = draw_horizontal_line_at_bottom_most_white(image, binary_map, last_green_line_x)

intersection_coordinates.insert(0, start)
intersection_coordinates.append(end)

print(intersection_coordinates)


new_image = draw_indicators(
    image, 
    binary_map, 
    intersection_coordinates, 
    draw_horizontal_lines=True,
    draw_vertical_lines=True,
    draw_intersection=True,
    draw_start_pnt=True,
    draw_end_pnt=True,
)

df = get_km_df_from_intersections(intersection_coordinates)
df.to_csv('kaplan_meier_data.csv', index=False)

plot_kaplan_meier(df)

# Save or display the result
cv2.imwrite(OUTPUT_IMAGE, new_image)

