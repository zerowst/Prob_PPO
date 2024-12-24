[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# ProbPPO:Exploring the Sampling Uncertainty for LLM Reasoning
> Songtao Wang and Zheng Xiong



<p align="center">
    <img src="assets/results.png" width="80%" />
</p>
*Generating up to 40K tok/sec on 8xH100 GPUs for 7B model

## Abstract
*Large Language Models (LLMs) have achieved remarkable progress in natural language processing (NLP), while they continue to face challenges in reasoning tasks due to their inherent complexity and consistency. Reinforcement Learning (RL) algorithms, such as Proximal Policy Optimization (PPO) and VinePPO, have shown potential for reasoning tasks but still remain constrained by inaccuracies in value estimation and inefficient utilization of sampled data. To address these limitations, we propose ProbPPO, a novel RL training framework designed to fully exploit data generated through Monte Carlo sampling. ProbPPO employs token prediction probabilities to represent the token generation space, thereby modeling the probability distribution of single sample across the entire sampling space. This approach allows more precise value estimation and advantage computation for each reasoning step. In addition, we introduce the importance sampling to acquire the weighted average rewards based on inference probability of each sample improving training stability and accuracy. We conducted experiment on mathematical reasoning tasks with the GSM8K dataset, demonstrating superior value estimation and reasoning accuracy of ProbPPO compared to state-of-the-art baselines. These results underscore the effectiveness of ProbPPO in enhancing reasoning capabilities of LLMs.*

## Quick Start

### Installation
This project is implemented based torch, Huggingface, FlashAttention, DeepSpeed, and vLLM libraries. To obtain the dependencies, we provide the following three ways:

**1. Using pip**
```bash
# Make sure torch 2.1.2 and cuda 12.1 is installed
pip install -r requirements.txt
```
**2. Using Docker**
```bash
sudo docker run \
  --ipc=host \
  --gpus all \
  kazemnejad/treetune:v15.1 \
  python -c "import torch; print(torch.__version__)"
```
*Optional: You can use the following [Dockerfile](https://github.com/McGill-NLP/VinePPO/blob/main/Dockerfile) to build your own image*

**3. Using Singularity Container**
```bash
singularity pull --arch amd64 library://realtreetune/dev/treetune:v15
singularity exec --nv treetune_v15.sif python -c "import torch; print(torch.__version__)"
```
### Download the datasets
```bash
chmod a+x scripts/download_and_prepare_dataset.sh
./scripts/download_and_prepare_dataset.sh
```

### Create Experiment Script

We first specify the configuration file for the experiment, and then, we explain how to run the training and evaluation using a configuration file.

**ProbPPO Experiments**
- `configs/polIter_rho1bSft2_vineppo_GSM8K.jsonnet`

**DPO Experiments**
- `configs/polIter_rho1bSft2_dpo_positive_GSM8K.jsonnet`

Once you have selected the configuration file, you can run the training and evaluation using the following script:
```bash

CONFIGSTR="configs/<config_file>.jsonnet"
APP_DIRECTORY="experiments/<path_to_output_dir>"

export APP_SEED="2746318213"
export WANDB_RUN_ID="<unique_wandb_run_id>" # Optional

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop

# Run the evaluation
deepspeed --no_local_rank --num_gpus=$NUM_GPUS   \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_evaluation

```

### Single GPU Training (Only for Rho models)
Add this config `configs/trainers/devBz16.jsonnet` to the `$CONFIGSTR` variable in the script above:
```bash
CONFIGSTR="configs/<config_file>.jsonnet,\
configs/trainers/devBz16.jsonnet"
```
Note that this is not fully tested and you may need to adjust the batch size to fit your GPU memory.

### Running the experiments
To run the experiments, you can use the following script:
1. Normal local run
```bash
chmod +x run.sh
./run.sh
```
2. Running inside docker
```bash
mkdir -p experiments
docker run \
    --ipc=host \
    --gpus all \
    -v "$(pwd)":/src \
    --workdir /src \
    kazemnejad/treetune:v15.1 \
    ./run.sh
```
3. Running inside singularity
```bash
mkdir -p experiments
chmod a+x run.sh
singularity exec --nv \
	-H $(pwd):$HOME \
	-B $(pwd)/experiments:$HOME/experiments \
	/path/to/singularity/image/treetune_v15.sif \
	./run.sh
```
## Initial SFT Checkpoints

|Base Model \ SFT Dataset                | GSM8K                                                                                  | MATH                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| DeepSeekMath 7B   | [🤗 Deepseekmath-SFT-GSM8K](https://huggingface.co/realtreetune/deepseekmath-7b-sft-GSM8K) | [🤗 Deepseekmath-SFT-MATH](https://huggingface.co/realtreetune/deepseekmath-7b-sft-MATH-v2) |
| RhoMath 1.1B          | [🤗 Rhomath-SFT-GSM8K](https://huggingface.co/realtreetune/rho-1b-sft-GSM8K)                 | [🤗 Rhomath-SFT-MATH](https://huggingface.co/realtreetune/rho-1b-sft-MATH)                 |

## Acknowledgement

This is the release codebase for ProbPPO. It is developed by [@zerowst](https://github.com/zerowst) and [@imadlak](https://github.com/ImadlakQvQ).

This codebase takes the whole part from the [VinePPO]([https://github.com/McGill-NLP/vineppo]).

### Important files
Trainers:
- [`ppo_trainer.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/trainers/ppo_trainer.py): The main PPO trainer which is shared between PPO and VinePPO.
- [`dpo_positive_trainer.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/trainers/dpo_positive_trainer.py): The DPO-Positive trainer.
- [`restem_trainer.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/trainers/restem_trainer.py): The RestEM trainer.

Episode Generators:
- [`math_episode_generator.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/episode_generators/math_episode_generator.py): The PPO episode generator.
- [`math_episode_generator_with_mc_advantages.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/episode_generators/math_episode_generator_with_mc_advantages.py): The VinePPO episode generator. This class contains the implementation for Monte Carlo value estimation.
- [`math_dpo_positive_episode_generator.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/episode_generators/math_dpo_positive_episode_generator.py): The DPO-Positive episode generator, which generate positive and negative pairs for DPO.
- [`math_restem_episode_generator.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/episode_generators/math_restem_episode_generator.py): The RestEM episode generator.

Tasks:
- [`math.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/tasks/math.py): The main task file for MATH dataset.
- [`gsm8k.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/tasks/gsm8k.py): The main task file for GSM8K dataset.
- [`math_grader_minerva.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/tasks/math_grader_minerva.py): The grader for MATH dataset.
- [`math_extract_steps_inplace`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/tasks/math_extract_steps_inplace.py): The helper script to split MATH-style solutions into steps.

Other:
- [`policy_iteration_runtime.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/runtime/policy_iteration_runtime.py): The main runtime script for running experiments including training and evaluation.
- [`vllm_server.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/common/vllm_server.py): The handler class for vLLM inference engine.
- [`cot_inference_strategy.py`](https://github.com/zerowst/Prob_PPO/tree/main/src/treetune/inference_strategies/cot_inference_strategy.py): The main class we use for running inferences with vLLM API.

## Most code work bsaed on VinePPO
```bibtex
@misc{Kazemnejad2024:VinePPO,
      title={VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment}, 
      author={Amirhossein Kazemnejad and Milad Aghajohari and Eva Portelance and Alessandro Sordoni and Siva Reddy and Aaron Courville and Nicolas Le Roux},
      year={2024},
      eprint={2410.01679},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.01679}, 
}
```
