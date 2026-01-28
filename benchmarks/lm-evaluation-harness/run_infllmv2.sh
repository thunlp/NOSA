# model_path=/home/test/test01/wpj/Megatron-LM/hf_ckpts/3b_infllmv2_sft
# model_path=/home/test/test01/hyx/Megatron-LM/hf_ckpts_1b_new_data_infllmv2_bugfix_sft/300
model_path=/home/test/test01/hyx/Megatron-LM/hf_checkpoints/8b_infllmv2_sft_llama/900 
export HF_ALLOW_CODE_EVAL=1
export HF_ENDPOINT=https://hf-mirror.com

accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks humaneval_instruct \
    --batch_size 32 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --apply_chat_template True \
    --log_samples \
    --output_path ./output/infllmv2 

accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mbpp \
    --batch_size 32 \
    --num_fewshot 3 \
    --confirm_run_unsafe_code \
    --output_path ./output/infllmv2 

accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu \
    --batch_size 32 \
    --num_fewshot 5 \
    --confirm_run_unsafe_code \
    --apply_chat_template True \
    --output_path ./output/infllmv2 

accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu_pro \
    --batch_size 32 \
    --num_fewshot 5 \
    --confirm_run_unsafe_code \
    --output_path ./output/infllmv2 

accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks bbh \
    --batch_size 32 \
    --num_fewshot 3 \
    --confirm_run_unsafe_code \
    --apply_chat_template True \
    --output_path ./output/infllmv2 


accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks gsm8k \
    --batch_size 32 \
    --num_fewshot 5 \
    --confirm_run_unsafe_code \
    --apply_chat_template True \
    --output_path ./output/infllmv2 

accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks hendrycks_math \
    --batch_size 32 \
    --num_fewshot 4 \
    --confirm_run_unsafe_code \
    --output_path ./output/infllmv2 


accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks drop \
    --batch_size 32 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --apply_chat_template True \
    --output_path ./output/infllmv2 