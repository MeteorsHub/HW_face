# HW_face

This project is based on mtcnn and facenet.

We achieve a wonderful result of top1 acc 98.05% and top5 acc 99.45%.

The results can be found in `predictions.txt`. Or you can follow the instructions below to reproduce training.

We offer trained model on [Clound Storage](https://cloud.tsinghua.edu.cn/d/39a2cf685c4841cbb8cb/). You can download it and skip sub-steps 1, 2, 3 in `Face model multi-train` step.

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

## Face model multi-train

If you want to train the model on your own, please follow the multi-train step below.

1. Download [pretrained model on vgg-face2](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) from [FaceNet's github](https://github.com/davidsandberg/facenet)
and save it to `saved_models/vgg_face2`
2. Fix Inception-V1 vars and only finetune the last fc layer on HW_Face dataset by running:
    ```bash
    python train_softmax.py \
    --logs_base_dir=logs/softmax_finetune \
    --models_base_dir=saved_models/vgg_face2_finetune_softmax/ \
    --pretrained_model=saved_models/vgg_face2/modelname \
    --data_dir=dataset/HW_1_Face/mtcnn_160/train \
    --test_dir=dataset/HW_1_Face/mtcnn_160/test \
    --max_nrof_epochs=100 \
    --gpu_memory_fraction=0.8 \
    --batch_size=100 \
    --epoch_size=20 \
    --embedding_size=512 \
    --learning_rate=0.1 \
    --learning_rate_decay_epochs=20 \
    --learning_rate_decay_factor=0.8 \
    --save_every_n_epochs=10 \
    --validate_every_n_epochs=1 \
    --keep_probability=0.5 \
    --weight_decay=0.01 \
    --center_loss_factor=1.0 \
    --center_loss_alfa=0.95 \
    --is_finetune \
    --load_without_fc
    
    ```
    Remember to change `--pretrained_model` to the one you download from github.
    
    This will lead to about top1 acc of 0.62 on test set.

3. Finetune all vars on HW_Face by running:
    ```bash
    python train_softmax.py \
    --logs_base_dir=logs/softmax_finetune \
    --models_base_dir=saved_models/vgg_face2_finetune_softmax/ \
    --pretrained_model=saved_models/vgg_face2_finetune_softmax/modelname \
    --data_dir=dataset/HW_1_Face/mtcnn_160/train \
    --test_dir=dataset/HW_1_Face/mtcnn_160/test \
    --max_nrof_epochs=100 \
    --gpu_memory_fraction=0.8 \
    --batch_size=300 \
    --epoch_size=20 \
    --embedding_size=512 \
    --learning_rate=0.01 \
    --learning_rate_decay_epochs=20 \
    --learning_rate_decay_factor=0.8 \
    --save_every_n_epochs=10 \
    --validate_every_n_epochs=1 \
    --keep_probability=0.5 \
    --weight_decay=0.001 \
    --center_loss_factor=2.0 \
    --center_loss_alfa=0.25 
    
    ```
    Remember to change `--pretrained_model` to the one last step generates.
    
    This will lead to about top1 acc of 0.82 on test set.
4. Retrieve via cosine distance
    Use `classifier_cos.py` to apply retrieve functions.
    ```bash
    python classifier_cos.py \
    dataset/HW_1_Face/mtcnn_160/train \
    dataset/HW_1_Face/mtcnn_160/test \
    saved_models/model_path_it_should_be \
    --batch_size=256 \
    --image_size=160
    ```
    Remember to change the 3rd argument to trained model path of last step or the one you have downloaded.
    
    This will generate `predictions_cos.txt` and at about top1 acc of 0.98.
    
    
## Comments
If you have any problems or something went wrong when training, feel free to contact me. Because this instruction and 
the code is summarized in a hurry and may contain something inaccurate.

> Instructions why we use test set for validation:
>
> There is no validation set in the provided data. And we only use validation for intermediate step result generation,
> not for hyper-parameters modulation. We adjust h-paras according to the loss. 
>
> The accuracy achieve 100% on training set very soon because of the small size of training set. So the primary 
> reference during training is the loss.