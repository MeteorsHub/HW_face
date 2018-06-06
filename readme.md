# HW_face

This project is based on mtcnn and facenet.

The results can be found in `predictions.txt`. Or you can follow the instructions below to perform test.

## Requirements

Tensorflow>=1.7

Python2.7 or 3.5

Other python packages you can install when running.

## Pre-process

set PYTHONPATH environment to this root directory like this on linux:
```bash
export PYTHONPATH=path/to/this/root/directory
```

## Prepare data

Move HW_1_Face data to `dataset/HW_1_Face/raw`

Change the data structure by running

```bash
python dataset/HW_1_Face/raw/sort.py
```

## Face detection

Using mtcnn to align face bounding box. Run train set:

```bash
python align/align_dataset_mtcnn.py \
dataset/HW_1_Face/raw/train/ \
dataset/HW_1_Face/mtcnn_160/train \
--image_size=160 \
--margin=32 \
--gpu_memory_fraction=0.85
```

and test set:

```bash
python align/align_dataset_mtcnn.py \
dataset/HW_1_Face/raw/test/ \
dataset/HW_1_Face/mtcnn_160/test \
--image_size=160 \
--margin=32 \
--gpu_memory_fraction=0.85
```

to get face detection of the data

## Face recognition

This will use pretrained checkpoint in `saved_models/vgg_face2` trained on vgg_face2

Download from [https://cloud.tsinghua.edu.cn/d/991128b4d19e43bf99fc/](https://cloud.tsinghua.edu.cn/d/991128b4d19e43bf99fc/)

We will use cosine distance to measure the similarity of test images with reference (train) images.


```bash
python classifier_cos.py \
dataset/HW_1_Face/mtcnn_160/train \
dataset/HW_1_Face/mtcnn_160/test \
saved_models/vgg_face2 \
--batch_size=256 \
--image_size=160
```

## Other method we tried

We use some other method but get worse result. They are `classifier_svm.py`, `finetune_cos.py` and `train_softmax.py`