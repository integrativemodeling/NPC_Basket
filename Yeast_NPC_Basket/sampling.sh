

export CURR_DIR=.
export DATA_DIR=./data
export OUT_DIR=./output
export SGE_TASK_ID=1

if [ ! -d $OUT_DIR/run_$SGE_TASK_ID ]; then
    mkdir $OUT_DIR/run_$SGE_TASK_ID
fi

cd $CURR_DIR/
pwd
python $CURR_DIR/src/modeling_basket.py -d $DATA_DIR -o $OUT_DIR/run_$SGE_TASK_ID/ > $OUT_DIR/run_$SGE_TASK_ID/npc_bskt_${SGE_TASK_ID}.log


