#!/bin/bash

mkdir data
mkdir data/person

wget https://pub.ist.ac.at/~mlechner/datasets/walker.zip
unzip walker.zip -d data/
rm walker.zip
