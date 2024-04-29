import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd 
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",offload_folder="offload",torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
guidelines=pd.read_csv('../../ICD-11 Guidelines.csv')

device = "cuda:0"

summary=[]
prefix=""" ### Instruction: Summarize this passage as short as possible
"""
suffix="""Answer:
"""
inputs=[prefix +' '+d+' \n'+g +'\n'+suffix for d,g in zip(guidelines['Correct Diagnosis'],guidelines['ICD11 Guidelines'])]
for i,n in enumerate(inputs): 
    print(i)
    encodeds = tokenizer(n, return_tensors="pt", add_special_tokens=True)
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids) 
    summary.append(decoded[0].replace("<s>", "").replace("</s>", "").split('Answer:',1)[1].strip())
    print(decoded[0].replace("<s>", "").replace("</s>", "").split('Answer:',1)[1].strip())
guidelines['summary']=summary
guidelines['summary'][8]=guidelines['ICD11 Guidelines'][8]
guidelines.to_csv('../../ICD-11 Guidelines.csv')
print(summary)
