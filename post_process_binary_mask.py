import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from skimage import img_as_bool, img_as_ubyte


def load_binary_mask(path, min_px_value=127):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, min_px_value, 255, cv2.THRESH_BINARY)
    return binary_mask


def skeletonize_binary_mask(binary_mask):
    binary_mask_bool = img_as_bool(binary_mask)
    skeleton = skeletonize(binary_mask_bool)

    # Convert back to uint8 for OpenCV (0 and 255 for binary)
    skeleton_uint8 = img_as_ubyte(skeleton)
    return skeleton_uint8


def dialate_binary_mask(binary_mask, dilate_kernel=5):
    # Step 3: Dilate the skeleton to achieve a max thickness of 5px
    kernel = np.ones((dilate_kernel,dilate_kernel), np.uint8)  # Create a kernel for dilation
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    return dilated


def smooth_binary_mask(binary_mask, blur_kernel=7):
    smoothed = cv2.GaussianBlur(binary_mask, (blur_kernel, blur_kernel), 0)
    return smoothed


def find_starting_ponit(binary_mask):
    # Find the starting point: topmost white pixel
    rows, cols = np.where(binary_mask == 255)
    return rows[0], cols[0]


def trace_line_over_binary_mask(binary_mask, max_gap=100):
    current_row, current_col = find_starting_ponit(binary_mask)

    new_binary_mask = np.zeros_like(binary_mask)

    # color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Trace the mask and draw the red line
    while current_row < binary_mask.shape[0] and current_col < binary_mask.shape[1]:
        # Draw red line at the current position
        
        new_binary_mask[current_row, current_col] = 255

        # Try to move horizontally to the right, if the next pixel is white
        if current_col + 1 < binary_mask.shape[1] and binary_mask[current_row, current_col + 1] == 255:
            current_col += 1
        # If no white pixel to the right, try to move vertically down
        elif current_row + 1 < binary_mask.shape[0] and binary_mask[current_row + 1, current_col] == 255:
            current_row += 1
        # Handle gaps: If no white pixels to the right or down, search in a window
        else:
            found_next_pixel = False
            
            # Search in a horizontal window for the next white pixel within max_gap
            for gap_col in range(1, max_gap + 1):
                if current_col + gap_col < binary_mask.shape[1] and binary_mask[current_row, current_col + gap_col] == 255:
                    current_col += gap_col
                    found_next_pixel = True
                    break

            # If no pixel found horizontally, search in a vertical window
            if not found_next_pixel:
                for gap_row in range(1, max_gap + 1):
                    if current_row + gap_row < binary_mask.shape[0] and binary_mask[current_row + gap_row, current_col] == 255:
                        current_row += gap_row
                        found_next_pixel = True
                        break

            # If no next pixel is found within the gap tolerance, stop
            if not found_next_pixel:
                break

    return new_binary_mask


def post_process_binary_mask(binary_mask, write_image=False):
    binary_mask = skeletonize_binary_mask(binary_mask)
    binary_mask = dialate_binary_mask(binary_mask)
    binary_mask = smooth_binary_mask(binary_mask)

    _, binary_mask = cv2.threshold(binary_mask, 230, 255, cv2.THRESH_BINARY)

    binary_mask = trace_line_over_binary_mask(binary_mask)

    if write_image:
        cv2.imwrite('post_processed_binary_mask.png', binary_mask)

    return binary_mask


def detect_events(binary_mask, min_vertical_px_length=3):
    # The input is assumed to be a 1px linewidth binary_map
    new_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    events = []

    _, start_x = find_starting_ponit(binary_mask)

    height, width = binary_mask.shape



    for x in range(start_x, width-1):
        log_px = []
        for y in range(height-1):
            if binary_mask[y, x] == 255:
                log_px.append((y, x))

        if len(log_px) >= min_vertical_px_length:
            log_y, log_x = log_px[0]
            events.append((log_x, log_y))
            new_image[log_y, log_x] = [0, 0, 255]
            for log_y, log_x in log_px:
                new_image[log_y, log_x] = [0, 0, 255]

        
    cv2.imwrite('find_vertical.png', new_image)
    
    return events


def get_kaplan_meier_data_from_events(events:list, group:int):
    times = [event[0] for event in events]
    number_at_risk = [event[1] for event in events]
    censoring_status = [group] * len(events)  # Assuming all events are observed (censoring status = 1)

    # Create a DataFrame
    df = pd.DataFrame({
        'Time': times,
        'Censoring_Status': censoring_status,
        'Number_at_Risk': number_at_risk
    })

    return df


if __name__ == "__main__":
    # binary_mask = load_binary_mask('clean_image/inst_mask_0.png')
    binary_mask = load_binary_mask('sample_result_mask_3.png')
    binary_mask = post_process_binary_mask(binary_mask, write_image=True)
    events = detect_events(binary_mask)

    df = get_kaplan_meier_data_from_events(events)

    df.to_csv('kaplan_meier_data.csv', index=False)
    print(events)

    
