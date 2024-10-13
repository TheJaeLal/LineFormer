import infer
import cv2
import line_utils

img_path = "demo/plt_0.png"
img = cv2.imread(img_path)

CKPT = "iter.pth"
CONFIG = "km_swin_t_config.py"
DEVICE = "cpu"

infer.load_model(CONFIG, CKPT, DEVICE)
line_dataseries, inst_masks = infer.get_dataseries(img, to_clean=False, return_masks=True)

# Visualize extracted line keypoints
img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))

cv2.imwrite("demo/sample_result.png", img)
for idx, inst_mask in enumerate(inst_masks):
    cv2.imwrite(f"demo/sample_result_mask_{idx}.png", inst_mask)
