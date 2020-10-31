#!/bin/bash
mkdir data 
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00570/clinc150_uci.zip -P ./data/.
unzip ./data/clinc150_uci.zip -d ./data/.
