# LineFormer - Rethinking Chart Data Extraction as Instance Segmentation

Official repository for the ICDAR 2023 Paper

[<u>[Link]</u>](https://link.springer.com/chapter/10.1007/978-3-031-41734-4_24) to the paper.

## Quantitative Results
| Dataset             | AdobeSynth19 Visual Element Detection[^1] | Data Extraction[^2] | UB-PMC22 Visual Element Detection | Data Extraction | LineEX Visual Element Detection | Data Extraction |
|---------------------|------------------------------------------|---------------------|----------------------------------|-----------------|---------------------------------|----------------|
| [ChartOCR](https://openaccess.thecvf.com/content/WACV2021/papers/Luo_ChartOCR_Data_Extraction_From_Charts_Images_via_a_Deep_Hybrid_WACV_2021_paper.pdf)        | 84.67                                    | 55                  | 83.89                            | 72.9            | 86.47                           | 78.25          |
| [Lenovo](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_37)          | **99.29**                                | **98.81**          | 84.03                            | 67.01           | -                               | -              |
| [LineEX](https://openaccess.thecvf.com/content/WACV2023/papers/P._LineEX_Data_Extraction_From_Scientific_Line_Charts_WACV_2023_paper.pdf)          | 82.52                                    | 81.97               | 50.23                         | 47.03           | 71.13                           | 71.08          |
| [**Lineformer**](https://arxiv.org/abs/2305.01837) (Ours)   | 97.51                                    | 97.02               | **93.1**                          | **88.25**       | **99.20**                       | **97.57**      |

[^1]: [task-6a from CHART-Info challenge](https://example.com/chart-info-task-6a)
[^2]: [task-6b data score from CHART-Info challenge](https://example.com/chart-info-task-6b)

<!-- **If you would like to cite our work:**
```latex

``` -->

## Model Usage
### Install Environment

This code is based on [MMdetection Framework](https://github.com/open-mmlab/mmdetection).

Code has been tested on Pytorch 1.13.1 and CUDA 11.7.

Create Conda Environment and install dependencies:
```bash
conda create -n LineFormer python=3.8
conda activate LineFormer
bash install.sh
```


### Inference

1. Download the Trained Model Checkpoint [here](https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T?usp=sharing)
2. Use the demo inference snippet shown below

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

## Citation
If you found our work useful, please cite us as follows:
```bib
@InProceedings{10.1007/978-3-031-41734-4_24,
author="Lal, Jay
and Mitkari, Aditya
and Bhosale, Mahesh
and Doermann, David",
editor="Fink, Gernot A.
and Jain, Rajiv
and Kise, Koichi
and Zanibbi, Richard",
title="LineFormer: Line Chart Data Extraction Using Instance Segmentation",
booktitle="Document Analysis and Recognition - ICDAR 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="387--400",
abstract="Data extraction from line-chart images is an essential component of the automated document understanding process, as line charts are a ubiquitous data visualization format. However, the amount of visual and structural variations in multi-line graphs makes them particularly challenging for automated parsing. Existing works, however, are not robust to all these variations, either taking an all-chart unified approach or relying on auxiliary information such as legends for line data extraction. In this work, we propose LineFormer, a robust approach to line data extraction using instance segmentation. We achieve state-of-the-art performance on several benchmark synthetic and real chart datasets. Our implementation is available at https://github.com/TheJaeLal/LineFormer.",
isbn="978-3-031-41734-4"
}
```

Note: LineFormer returns data in form of x,y points w.r.t the image, to extract full data-values you need to extract axis information, which can be done using [this](https://github.com/pengyu965/ChartDete/) repo.
