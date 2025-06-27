from transformers.pipelines import pipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.constants import TRAINING_MODEL

LORA_WEIGHTS_PATH = "./lora-model"

def load_finetuned_llm():
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(TRAINING_MODEL, device_map="auto")
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
    model.eval()
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
    )

    return text_gen_pipeline  
