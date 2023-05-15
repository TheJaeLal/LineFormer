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
```python
import infer
import cv2

img_path = "demo/sample_line.png"
img = cv2.imread(img_path) # BGR format
line_dataseries = infer.get_dataseries(img, to_clean=False)
```

Example extraction result:

![demo result](sample_result.jpg "Title")
