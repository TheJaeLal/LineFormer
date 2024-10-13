import infer
import cv2
import line_utils

# img_path = "demo/PMC5959982___3_HTML.jpg"
img_path = "demo/plt_0.png"
img = cv2.imread(img_path)  # BGR format

# CKPT = "iter_3000.pth"
CKPT = "iter_11500.pth"
CONFIG = "lineformer_swin_t_config.py"
# CONFIG = "km_swin_t_config.py"
DEVICE = "cpu"

infer.load_model(CONFIG, CKPT, DEVICE)
line_dataseries, inst_masks = infer.get_dataseries(img, to_clean=False, return_masks=True)

# Visualize extracted line keypoints
img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))

cv2.imwrite("demo/sample_result.png", img)
cv2.imwrite("demo/sample_result_mask.png", inst_masks)
