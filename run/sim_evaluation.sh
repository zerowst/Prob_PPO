CONFIGSTR="configs/polIter_rho1bSft2_vineppo_GSM8K.jsonnet"
export APP_DIRECTORY="sim_checkpoints/"

export APP_SEED="2746318213"
export WANDB_RUN_ID="12012" # Optional


NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# # Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_evaluation

