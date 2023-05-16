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
import matplotlib.pyplot as plt
import line_utils

img_path = "demo/sample_line.png"
img = cv2.imread(img_path) # BGR format
line_dataseries = infer.get_dataseries(img, to_clean=False)

# Visualize extracted line keypoints
plt.imshow(line_utils.draw_kps(img, line_dataseries))

```

Example extraction result:

![demo result](demo/sample_result.jpg "Title")

Note: LineFormer returns data in form of x,y points w.r.t the image, to extract full data-values you need to extract axis information, which can be done using [this](https://github.com/pengyu965/ChartDete/) repo.
