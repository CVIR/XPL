#!/bin/bash

# custom config
DATA= ./datasets/enprompt
TRAINER=XPL

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
TYPE=$7 # Either shot or pt
TH=0.7
MU=7
LR=0.005

for SEED in 1 2 3
do
    DIR=output_XPL/${DATASET}/${CFG}_${SHOTS}${TYPE}/seed${SEED}
    if [ -d "$DIR" ]; then 
        echo "Oops! The results exist at ${DIR} (so skip this job)" 
	else
	    python train.py \
	    --root ./datasets/enprompt \
	    --seed ${SEED} \
	    --trainer ${TRAINER} \
	    --dataset-config-file configs/datasets/${DATASET}.yaml \
	    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
	    --output-dir ${DIR} \
	    --th ${TH} \
	    --mu ${MU} \
	    --lr ${LR} \
	    --resume ${DIR} \
	    TRAINER.COOP.N_CTX ${NCTX} \
	    TRAINER.COOP.CSC ${CSC} \
	    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
	    DATASET.NUM_SHOTS ${SHOTS} \
	    TYPE ${TYPE}
    fi
done