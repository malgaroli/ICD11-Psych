from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from tqdm import tqdm
import logging
import gc

logging.basicConfig(level=logging.INFO)

class LLMModel:
    
    def __init__(self, model_path, use_gpu=True,  max_new_tokens=512):
        self._max_new_tokens = max_new_tokens
        HF_TOKEN = "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"
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN) #, use_fast=True)
        self._eos_token_id = self.tokenizer.eos_token_id
        self._is_gpt_oss = "gpt-oss" in (str(model_path) or "").lower()

        # Ensure a neutral pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self._eos_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')

        if use_gpu:
            if self._is_gpt_oss:
                # GPT-OSS path: no accelerate, no bnb; rely on native dtype + device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    token=HF_TOKEN,
                    torch_dtype=torch.bfloat16, # "auto" lets HF pick bf16 on capable GPUs
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                # from transformers import BitsAndBytesConfig
                # # Quantization config with correct compute dtype
                # bnb_config = BitsAndBytesConfig(
                #     load_in_4bit=True,
                #     bnb_4bit_compute_dtype=torch.float16,
                #     llm_int8_skip_modules=None  # Optional: skip quantization for specific modules
                # )

                # # Load quantized model
                # self.model = AutoModelForCausalLM.from_pretrained(
                #     model_path,
                #     device_map="auto",
                #     quantization_config=bnb_config,
                #     torch_dtype=torch.float16,
                #     low_cpu_mem_usage=True,
                #     token=HF_TOKEN
                # )
            else:
                from accelerate import Accelerator
                from transformers import BitsAndBytesConfig

                # Initialize accelerator
                self.accelerator = Accelerator(mixed_precision="bf16")

                # Quantization config with correct compute dtype
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_skip_modules=None  # Optional: skip quantization for specific modules
                )

                # Load quantized model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    token=HF_TOKEN
                )

                # Prepare model with accelerator
                self.model = self.accelerator.prepare(self.model)

        else:
            # Load model for CPU inference
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
    def _get_assistant_marker(self):
        # Define per tokenizer/model
        name = self.tokenizer.name_or_path.lower()
        if "llama" in name:
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "deepseek" in name:
            #added this for deepSeek
            return "</think>\n\n"
        elif "mistral" in name:
            return "[/INST] "  # example for some chat templates
        elif "gemma" in name:
            return "<start_of_turn>model\n"
        elif "qwen" in name:
            return "<|im_start|>assistant\n"
        elif "gpt" in name:
            return "<|start|>assistant<|channel|>final<|message|>" 
        else:
            raise ValueError("Unknown tokenizer template for assistant marker.")

    def _generate_response(self, prompts, do_sample=False, temperature=0.0, top_p=0.0):
        """
        Generates responses for a list of prompts.
        """
        # Format prompt for different models
        formatted_prompts = [self._format_prompt(p) for p in prompts]

        if len(formatted_prompts) > 1:
            inputs = self.tokenizer(formatted_prompts, return_tensors='pt',padding_side='left', padding=True, truncation=True).to(self.model.device)
        else:
            inputs = self.tokenizer(formatted_prompts, return_tensors='pt',padding=False, truncation=True).to(self.model.device)


        gen_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._eos_token_id,
            "do_sample": do_sample,
        }
        
        if do_sample:
            # only set parameters in non-deterministic mode
            gen_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
            })

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        decoded_sequences = self.tokenizer.batch_decode(outputs)
        assistant_marker = self._get_assistant_marker()

        assistant_responses = []
        logging.info(f"Output length: {len(decoded_sequences[0])}")
        for sequence in decoded_sequences:
            parts = sequence.split(assistant_marker)
            assistant_text = parts[1] if len(parts) > 1 else parts[0]
            cleaned_text = assistant_text.replace(self.tokenizer.pad_token, "").replace("<|eot_id|>", "")
            assistant_responses.append(cleaned_text)

        return assistant_responses
    
    
    def _format_prompt(self, messages_or_text):
        if isinstance(messages_or_text, str):
            return messages_or_text
        elif isinstance(messages_or_text, list) and all("role" in message and "content" in message for message in messages_or_text):
            try:
                if "deepseek" in self.tokenizer.name_or_path.lower():
                    # For deepSeek there is no system prompt allowed -> combine system with user prompt 
                    # Extract system and user messages, if any
                    system_msg = next((m["content"] for m in messages_or_text if m["role"] == "system"), "").strip()
                    user_msg = next((m["content"] for m in messages_or_text if m["role"] == "user"), "").strip()

                    # Combine system + user content into a single user message
                    combined_content = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
                    messages_or_text = [{"role": "user", "content": combined_content}]

                return self.tokenizer.apply_chat_template(
                    messages_or_text,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                raise ValueError("Tokenizer does not support chat templates.") from e
        else:
            raise ValueError("Invalid input format for prompt.")


    def process_prompt(self, prompt, id):
        """
        Process a single prompt.
        """
        try:
            response = self._generate_response([prompt])[0]
            return {id: response}
        except Exception as e:
            logging.error(f"Error processing ID {id}: {e}")
            return {id: None}

    def _process_batch(self, batch):
        """
        Process a batch of inputs.
        """
        prompts = [p['prompt'] for p in batch]
        ids = [p['id'] for p in batch]
        try:
            responses = self._generate_response(prompts)
            return [{id_: resp} for id_, resp in zip(ids, responses)]
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
            return [{id_: None} for id_ in ids]

    def _batch_processor(self, data, batch_size):
        """
        Generator for processing in batches.
        """
        for i in tqdm(range(0, len(data), batch_size)):
            yield self._process_batch(data[i:i + batch_size])

    def process_all_batches(self, prompts, batch_size):
        """
        Run inference over all prompts in batches.
        """
        results = []
        for batch in self._batch_processor(prompts, batch_size):
            results.extend(batch)
        return results
