CONFIGSTR="configs/polIter_rho1bSft2_vineppo_MATH.jsonnet"
APP_DIRECTORY="experiments"


export APP_SEED="2746318213"
export WANDB_RUN_ID="1120" # Optional


NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


# debug the training
export DEBUG=1
deepspeed --no_local_rank --num_gpus=$NUM_GPUS \
    src/treetune/main.py --configs "$CONFIGSTR" \
    run_iteration_loop --master_port=5678