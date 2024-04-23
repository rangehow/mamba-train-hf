accelerate launch --config_file config.yaml mamba-train-hf/train.py 

如果没有装deepspeed

torchrun --master-port 31451 --nproc-per-node 4 mamba-train-hf/train.py --lora