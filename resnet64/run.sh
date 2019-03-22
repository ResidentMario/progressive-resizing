#!/bin/bash
if [[ "$1" = train ]]
then
    alekseylearn fetch ./ resnet48/train s3://alpha-quilt-storage/aleksey/progressive-resizing/
    aws s3 cp s3://quilt-example/quilt/open_fruit/training_data/X_meta.csv X_meta.csv
    aws s3 sync s3://quilt-example/quilt/open_fruit/images_cropped/ images_cropped/
    python train.py
fi