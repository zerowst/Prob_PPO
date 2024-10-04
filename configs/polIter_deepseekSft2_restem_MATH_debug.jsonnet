local hf_model_name = 'realtreetune/deepseekmath-7b-sft-MATH-v2';

local math_task = (import 'tasks/math_inplace_no_answer_prefix.jsonnet') + {
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

local num_rollouts_per_sample = 8;
local total_num_iterations = 10;
local sampling_temperature = 0.6;

# taken from sft_deepseekmath_for_MATH_eval.jsonnet
local math_inference_pipeline =
    (import 'prompt_library/generic_MATH_step_by_step.jsonnet')
    + (import 'inference_strategies/tree/iid_expander.jsonnet')
    + (import 'inference_strategies/cot.jsonnet')
    + {
        inference_strategy+: {
            max_concurrent_programs: 128,
            max_concurrent_generations: 64,

            node_expander+: {
                type: 'efficient_iid',
                program_kwargs: {
                    temperature: 0.35,
                    top_p: 0.9,
                    max_tokens: 1024,
                    stop: '"\n\n\nProblem:"',
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: 4095,
                tokenizer: {
                    type: 'pretrained',
                    hf_model_name: hf_model_name,
                },
            },
            answer_extractor+: {
                type: 'identity',
                node_key_name: 'text',
            },
            samples: 16,
            max_depth: 10,

            guidance_llm: (import 'guidance_llms/deepseekmath7b-sft-MATH-v2.jsonnet') + { api_base: 'none' },
            no_cache: true,
            question_field: 'query',

            seed: 42,
        },
        task: (import 'tasks/math_inplace_no_answer_prefix.jsonnet'),
        analyzers: [(import 'analyzers/task_performance.jsonnet')],
    };

local math_validation_inference_pipeline =
     math_inference_pipeline
     + {
         dataset_split: 'validation',
         dataset_portion: 1,
         inference_name: 'math_validation',
     };

(import 'gvar.jsonnet')
+ (import 'prompt_library/MATH_step_by_step_sft.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'episode_generators/math_episode_generator.jsonnet')
+ (import 'trainers/restem_MATH.jsonnet')
+ {
    episode_generator+: {
        vllm_server+: {
            swap_space: 8,
        },
        // Override the task
        type: "math_restem_episode_generator",
        reward_threshold: 1.0,
        task: math_task,
        reward_function+: { math_task: $.episode_generator.task },
        reasoning_step_delimiter: '',
        answer_prefix: null,

        initial_model_name_or_path: hf_model_name,

        dataset_sample_with_replacement: false,
        dataset_portion: 0.01,
        total_num_iterations: total_num_iterations,

        max_sequence_length: 2499,
        save_generations_every_n_iteration: 50,

        inference_strategy: {
            type: 'cot',

            max_concurrent_programs: 128,
            max_concurrent_generations: 64,

            samples: num_rollouts_per_sample,
            max_depth: 100,  // Deprecated parameter. Doesn't do anything.

            node_expander: {
                type: 'efficient_iid',
                program: $.prompt_library.tree.expansion.iid,
                program_kwargs+: {
                    temperature: sampling_temperature,
                    top_p: 0.9,
                    max_tokens: 1024,
                    stop: '"\n\n\nProblem:"',
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: 4095,
                tokenizer: $.tokenizer,
            },

            answer_extractor: {
                type: 'identity',
                node_key_name: 'text',
            },

            guidance_llm: (import 'guidance_llms/deepseekmath7b-sft-MATH-v2.jsonnet') + { api_base: 'none' },

            question_field: 'query',
            question_template: $.prompt_library.tree.question_template,

            no_cache: false,
        },
    },

    tokenizer: {
        type: 'pretrained',
        hf_model_name: $.episode_generator.initial_model_name_or_path,
    },
    use_deepspeed: true,

    num_iterations: total_num_iterations,
    num_episodes_per_iteration: null,
    episodes_cloud_log_steps: 50,

    trainer+: {
        type: "restem",
        sampling_temperature: sampling_temperature,
        num_epochs_per_iteration: 8,
        actor_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },
        reference_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path},

        general_training_args+: {
            save_steps: 500,
            checkpoint_keep_steps: 500,
        },

        kl_penalty_loss_type: 'control_variate',

        cache_deepspeed_engines: false,  # because it OOM on mila cluster, we can try to cache the engines on microsoft cluster
        move_reference_model_to_cpu: false, # because it OOM on mila cluster, we can try to cache the engines on microsoft cluster

        # this is needed for the early stopping based on performance on validation split
        early_stop_tokenizer: $.tokenizer,
        early_stop_vllm_server: { swap_space: 8},
        early_stop_inference_pipeline_cfg: math_validation_inference_pipeline  +
        {
            inference_strategy+: {
                    node_expander+: {
                        tokenizer: $.tokenizer,
                },
            },
        },
    },

    analyzers: [
    (import 'analyzers/restem_upload_chosen_checkpoint.jsonnet'),
    (import 'analyzers/kl_with_reference.jsonnet') + {
            task: $.episode_generator.task,
            inference_strategy +: $.episode_generator.inference_strategy + {
                samples: num_rollouts_per_sample,
                },
            tokenizer: $.tokenizer,
            reward_function: $.episode_generator.reward_function,
            append_bos_to_query: $.episode_generator.append_bos_to_query,
            append_eos_to_response: $.episode_generator.append_eos_to_response,
        },
    ],
}
 + (import 'sft_deepseekmath_for_MATH_eval.jsonnet')