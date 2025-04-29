from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
import torch
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class LLMModel:
    def __init__(self, model_path, use_gpu=True,  max_new_tokens=512):
        self._max_new_tokens = max_new_tokens

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self._eos_token_id = self.tokenizer.eos_token_id

        # Ensure a neutral pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self._eos_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')

        if use_gpu:
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
                low_cpu_mem_usage=True
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
        if "llama" in self.tokenizer.name_or_path.lower():
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "mistral" in self.tokenizer.name_or_path.lower():
            return "[/INST] "  # example for some chat templates
        else:
            raise ValueError("Unknown tokenizer template for assistant marker.")

    def _generate_response(self, prompts, do_sample=False, temperature=0.0, top_p=0.0):
        """
        Generates responses for a list of prompts.
        """
        # Format prompt for different models
        formatted_prompts = [self._format_prompt(p) for p in prompts]
        inputs = self.tokenizer(formatted_prompts, return_tensors='pt',padding_side='left', padding=True, truncation=True).to(self.model.device)


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

        decoded = self.tokenizer.batch_decode(outputs)
        
        assistant_marker = self._get_assistant_marker()

        assistant_output = [decoded[i].split(assistant_marker)[1].replace(self.tokenizer.pad_token,"") for i in range(0,len(decoded))]
        return assistant_output
    
    
    def _format_prompt(self, messages_or_text):
        if isinstance(messages_or_text, str):
            return messages_or_text
        elif isinstance(messages_or_text, list) and all("role" in message and "content" in message for message in messages_or_text):
            try:
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
