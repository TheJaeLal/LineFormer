import infer
import pandas as pd
import cv2
import line_utils
import post_process_prediction.post_process_utils as post_process_utils
from post_process_prediction.extract_events import extract_events


# from post_process_binary_mask import post_process_binary_mask, detect_events, get_kaplan_meier_data_from_events

CKPT = "iter.pth"
CONFIG = "km_swin_t_config.py"
DEVICE = "cpu"

def run_inference(img):
    line_dataseries, inst_masks = infer.get_dataseries(img, to_clean=False, return_masks=True)

    # Visualize extracted line keypoints
    prediction_image = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))
    all_df = []
    for idx, inst_mask in enumerate(inst_masks):
        df = extract_events_df(inst_mask, group_idx=idx, write_debug=True, map_to_plot_coordinates=False)
        all_df.append(df)

    kaplan_meier_df = pd.concat(all_df, ignore_index=True)
    return prediction_image, inst_masks, kaplan_meier_df


if __name__ == '__main__':
    infer.load_model(CONFIG, CKPT, DEVICE)

    img_path = "demo/plt_0.png"
    img = cv2.imread(img_path)

    prediction_image, inst_masks, kaplan_meier_df = run_inference(img)
    cv2.imwrite("demo/sample_result.png", prediction_image)
    kaplan_meier_df.to_csv('demo/kaplan_meier_data.csv', index=False)

    for idx, inst_mask in enumerate(inst_masks):
        cv2.imwrite(f"demo/sample_result_mask_{idx}.png", inst_mask)
