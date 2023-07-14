#!/bin/bash
cd $HOME/--/

source ../Progressive\ Transformers/transformer-env/bin/activate

python main/get_predictions.py $1 --data $2 --batch_size 128 -wm quarter
python main/get_predictions.py $1 --data $2 --batch_size 128 -wm half
python main/get_predictions.py $1 --data $2 --batch_size 128 -wm full



