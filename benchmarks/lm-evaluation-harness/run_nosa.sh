model_path="openbmb/NOSA-8B"

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
    --output_path ./output/nosa 

accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mbpp \
    --batch_size 32 \
    --num_fewshot 3 \
    --confirm_run_unsafe_code \
    --output_path ./output/nosa 

accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu \
    --batch_size 32 \
    --num_fewshot 5 \
    --confirm_run_unsafe_code \
    --output_path ./output/nosa 

accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu_pro \
    --batch_size 32 \
    --num_fewshot 5 \
    --confirm_run_unsafe_code \
    --output_path ./output/nosa 


accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks bbh \
    --batch_size 32 \
    --num_fewshot 3 \
    --confirm_run_unsafe_code \
    --apply_chat_template True \
    --output_path ./output/nosa 


accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks gsm8k \
    --batch_size 32 \
    --num_fewshot 5 \
    --confirm_run_unsafe_code \
    --apply_chat_template True \
    --output_path ./output/nosa 


accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks hendrycks_math \
    --batch_size 32 \
    --num_fewshot 4 \
    --confirm_run_unsafe_code \
    --output_path ./output/nosa 


accelerate launch lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks drop \
    --batch_size 32 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --apply_chat_template True \
    --output_path ./output/nosa 