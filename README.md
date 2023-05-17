# LineFormer - Rethinking Chart Data Extraction as Instance Segmentation

Official repository for the ICDAR 2023 Paper

[<u>[Link]</u>](https://arxiv.org/abs/2305.01837) to the paper.

<!-- **If you would like to cite our work:**
```latex

``` -->

## Model Usage
### Install Environment

This code is based on [MMdetection Framework](https://github.com/open-mmlab/mmdetection) The following is brief environment and installation:

Code has been tested on Pytorch 1.13.1 and CUDA 11.7.

Create Conda Environment and install dependencies:
```bash
conda create -n LineFormer python=3.8
conda activate LineFormer
bash install.sh
```


### Inference

1. Download the Trained Model Checkpoint [here](https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T?usp=sharing)
2. Verify the Path to Pretrained Checkpoint in `lineformer_swin_t_config.py` and `infer.py`
3. Use the demo inference snippet shown below

```python
import infer
import cv2
import line_utils

img_path = "demo/PMC5959982___3_HTML.jpg"
img = cv2.imread(img_path) # BGR format

CKPT = "iter_3000.pth"
CONFIG = "lineformer_swin_t_config.py"
DEVICE = "cpu"

infer.load_model(CONFIG, CKPT, DEVICE)
line_dataseries = infer.get_dataseries(img, to_clean=False)

# Visualize extracted line keypoints
img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))
    
cv2.imwrite('demo/sample_result.png', img)


```

Example extraction result:

![input image](demo/PMC5959982___3_HTML.jpg "Input")
![demo result](demo/sample_result.png "Detection Result")

Note: LineFormer returns data in form of x,y points w.r.t the image, to extract full data-values you need to extract axis information, which can be done using [this](https://github.com/pengyu965/ChartDete/) repo.
