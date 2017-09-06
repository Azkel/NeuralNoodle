#! /bin/sh

if [ ! -f gogh.jpg ]; then
    wget https://smyk.it/files/gogh.jpg
fi

if [ ! -f vgg19_weights.h5 ]; then
    wget https://smyk.it/files/vgg19_weights.h5
fi


if [ ! -f test ]; then
    mkdir test
fi
