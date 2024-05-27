from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration

class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self, inference_only):
        if self.model is None:
            if "flan-ul2" in self.model_name:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name, device_map="auto", torch_dtype=th.bfloat16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="auto", torch_dtype=th.bfloat16
                )
            if inference_only:
                self.model.eval()
        return self.model

    def load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def save_model(self, output_dir):
        if self.model:
            self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

    def load_finetuned_model(self, output_dir):
        if "flan-ul2" in self.model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(output_dir)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
        return self.model, self.tokenizer
