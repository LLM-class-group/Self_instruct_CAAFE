export MODEL_NAME="llama3_lora_sft_v2"
export DATA_DIR="data"
export DATA_NAME="caafe_finetuning_data"
export BASE_MODEL="/home/jiahe/model_cache/LLM-Research/Meta-Llama-3-8B-Instruct"

cd /home/jiahe/ML/Self_instruct_CAAFE/LLaMA-Factory

CUDA_VISIBLE_DEVICES=0 python \
    src/train.py \
    --stage sft \
    --do_train True \
    --model_name_or_path ${BASE_MODEL} \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --template llama3 \
    --dataset_dir ${DATA_DIR} \
    --dataset ${DATA_NAME} \
    --cutoff_len 1120 \
    --learning_rate 0.0001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --preprocessing_num_workers 16 \
    --max_steps 1600 \
    --save_steps 400 \
    --warmup_steps 100 \
    --output_dir checkpoints/${MODEL_NAME} \
    --fp16 True \
    --plot_loss True \
    --overwrite_output_dir