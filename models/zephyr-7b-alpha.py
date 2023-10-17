import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import torch
from transformers import pipeline


# Loading the .env environment variables
load_dotenv()
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", device_map="auto")

# We use the tokenizer's chat template to format each message - see
# https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
