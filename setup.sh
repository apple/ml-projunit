#!/usr/bin/env bash
conda install python=3.7

python3 get-pip.py

pip3 install --upgrade pip
pip3 install --upgrade turibolt --index https://pypi.apple.com/simple
pip3 install -U numpy
pip3 install -U scipy
pip3 install -U torch
pip3 install -U torchvision
pip3 install -U boto3
pip3 install --upgrade awscli
pip3 install torchtext
pip3 install spacy
pip3 install sparse
pip3 install opacus
pip3 install kymatio
pip3 install seaborn


