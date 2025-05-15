# Readme

### Step 1: Generate TMM dataset
Run the script to Generate the thin film structure and reflectance value data and save them as a csv file
```
python TMM.py
```

### Step 2: Generate Alpaca json
Put the csv file into the right path
Run the script to Generate to turn the raw data into a trainable form and save them as a json file
```
python GAI.py
```

### Step 3: Finetune the LLM
Use the llama factory to finetune the model the training details are:
```
llamafactory-cli train `
    --stage sft `
    --do_train True `
    --model_name_or_path D:\llm_models\qwen3-0.6b `
    --preprocessing_num_workers 16 `
    --finetuning_type full `
    --template qwen3 `
    --flash_attn auto `
    --dataset_dir data `
    --dataset alpaca_en_demo `
    --cutoff_len 2048 `
    --learning_rate 5e-05 `
    --num_train_epochs 10.0 `
    --max_samples 100000 `
    --per_device_train_batch_size 2 `
    --gradient_accumulation_steps 8 `
    --lr_scheduler_type cosine `
    --max_grad_norm 1.0 `
    --logging_steps 5 `
    --save_steps 100 `
    --warmup_steps 0 `
    --packing False `
    --report_to none `
    --output_dir saves\Qwen2.5-0.5B-Instruct\full\train_2025-05-15-16-09-58 `
    --bf16 True `
    --plot_loss True `
    --trust_remote_code True `
    --ddp_timeout 180000000 `
    --include_num_input_tokens_seen True `
    --optim adamw_torch
```
After the training is done use it evaluate function to get predicted values

### Step 4: evaluate the model

Put the predicted value json into the right path
Run the script to run TMM again to draw the predicted plot
```
python eval.py
```





