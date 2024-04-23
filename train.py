from transformers import (
    MambaForCausalLM,
    Trainer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
import argparse
import torch
import os
import datasets
from functools import partial
from peft import LoraConfig,get_peft_model

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="state-spaces/mamba-2.8B-hf")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--optim", default="adamw_bnb_8bit")
    parser.add_argument("--dataset", default="vicgalle/alpaca-gpt4")
    return parser.parse_args()


def alpaca(instances, tokenizer):
    instruction, input, output = (
        instances["instruction"],
        instances["input"],
        instances["output"],
    )
    real_input = [ins + inp for ins, inp in zip(instruction, input)]

    def reformate(i, o):
        if o is not None:
            chat_dict = [
                {"role": "user", "content": i},
                {"role": "assistant", "content": o},
            ]
            return tokenizer.apply_chat_template(chat_dict)
        else:
            chat_dict = [
                {"role": "user", "content": i},
            ]
            return tokenizer.apply_chat_template(chat_dict,add_generation_prompt=True)
    
    input_ids,labels=[],[]
    
    for i,o in zip(real_input,output):
        chat_text=reformate(i,o)
        input_for_length=len(reformate(i,None))
        label=[-100]*input_for_length+chat_text[input_for_length:]
        input_ids.append(chat_text)
        labels.append(label)
    return {'input_ids':input_ids,'labels':labels}


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    tokenizer.chat_template = AutoTokenizer.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta"
    ).chat_template

    dataset = datasets.load_dataset(args.dataset)['train']
    # print(dataset)
    train_dataset = dataset.map(partial(alpaca, tokenizer=tokenizer), batched=True,num_proc=16,remove_columns=dataset.features.keys())
    model=MambaForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    collator=DataCollatorForSeq2Seq(tokenizer,model)
    # dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=2,collate_fn=collator)
    # for d in dataloader:
    #     print(d)
    #     import pdb
    #     pdb.set_trace()
    if args.lora:
        lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none",
        use_rslora=True,
        )
        model = get_peft_model(model, lora_config)
        
    trainer=Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            optim=args.optim,
            output_dir='mamba-out',
            save_strategy='epoch',
            bf16=True,
            remove_unused_columns=True,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            dataloader_num_workers=8,
        )
        
    )
    trainer.train()
    trainer.model.merge_and_unload(safe_merge=True)
    trainer.save_model('mamba-out')
    
