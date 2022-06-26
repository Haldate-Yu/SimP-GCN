#!/bin/bash
python ./src/train_simp.py \
--dataset citeseer \
--ptb_rate 0 \
--bias_init 2 \
--k 20 \
--gamma 1 \
--lambda_ 100 \
--seed 15 \
--epochs 200 \
--lr 0.01 \
--hidden 128 \
--weight_decay 5e-04 \
--ssl ECTDSim \
--datapath data// \
--type mutigcn \
--nhiddenlayer 1 \
--nbaseblocklayer 0 \
--early_stopping 200 \
--sampling_percent 1 \
--dropout 0.5 \
--normalization AugNormAdj --task_type semi \
\
