from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLLM:
    def __init__(self, device: str = "cuda"):
        self.device = device
        model_id = "Qwen/Qwen3-4B-Instruct-2507-FP8"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True
        )

    def chat(self, messages: list[dict], **kwargs) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(input_ids, **kwargs)

        return self.tokenizer.decode(
            output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
