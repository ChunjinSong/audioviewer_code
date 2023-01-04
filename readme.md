# AudioViewer: Learning to Visualize Sounds (WACV 2023)
 
![](img/teaser.jpg)

> **AudioViewer: Learning to Visualize Sounds** <br>
> [Chunjin Song*](https://chunjinsong.github.io/), Yuchi Zhang*, Willis Peng, Parmis Mohaghegh, [Bastian Wandt](https://bastianwandt.de/), and [Helge Rhodin](https://www.cs.ubc.ca/~rhodin/web/) <br>

[[Paper](https://arxiv.org/pdf/2012.13341.pdf)][[Website](https://chunjinsong.github.io/audioviewer/)]

## Setup

### Requirements

- Librosa >= 0.8
- Python >= 3.7
- PyTorch >= 1.5
- CUDA enabled computing device

### Dependencies

Install all the Python dependencies using pip:

~~~
pip install -r requirements.txt
~~~

Alternatively, you can install them with Anaconda:

~~~
conda install -n <env_name> requirements.txt
~~~
## Dataset
Dataset can be found from [google drive](black)

## Pre-trained models
Pre-trained models can be found from [google drive](black)

## Training

To train the models, please refer to the corresponding notebooks in `trainint.py`

## Evaluate

To evaluate the trained models, please refer to `testing.py`

## Citation

```
@article{audioviewer,
          title={AudioViewer: learning to visualize sound},
          author={Chunjin, Song and Zhang, Yuchi and Peng, Willis and Wandt, Bastian and Rhodin, Helge},
          journal={WACV},
          year={2023}
        }
```
