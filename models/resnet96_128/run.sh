#!/bin/bash
if [[ "$1" = train ]]
then
    aws s3 cp s3://quilt-example/quilt/open_fruit/training_data/X_meta.csv \
        X_meta.csv --quiet
    aws s3 sync s3://quilt-example/quilt/open_fruit/images_cropped/ \
        images_cropped/ --quiet
    aws s3 cp \
        s3://alpha-quilt-storage/aleksey/progressive-resizing/resnet48-128-train-1/output/model.tar.gz \
        ./model.tar.gz --quiet
    mkdir resnet48_128
    tar -xzf model.tar.gz -C resnet48_128
    rm model.tar.gz
    rm -rf 'images_cropped/Mango'
    rm -rf 'images_cropped/Common_fig'
    rm -rf 'images_cropped/Cantaloupe'
    rm -rf 'images_cropped/Pomegranate'
    python train.py
fi
