#!/bin/bash
if [[ "$1" = train ]]
then
    aws s3 cp s3://quilt-example/quilt/open_fruit/training_data/X_meta.csv \
        X_meta.csv --quiet
    aws s3 sync s3://quilt-example/quilt/open_fruit/images_cropped/ \
        images_cropped/ --quiet
    rm -rf 'images_cropped/Mango'
    rm -rf 'images_cropped/Common_fig'
    rm -rf 'images_cropped/Cantaloupe'
    rm -rf 'images_cropped/Pomegranate'
    python train.py
fi
