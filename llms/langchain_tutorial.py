import os
import replicate

# Zhen's hf token
HUGGINGFACE_API_TOKEN = "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"
os.environ['HUGGINGFACE_API_TOKEN'] = HUGGINGFACE_API_TOKEN

from langchain import PromptTemplate
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.9})


prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template="Write an audio ad script for new product called {product} that targets {audience}?",
)

print(prompt.format(product="xphone", audience="black young kids"))

llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain({"product": "xphone", "audience": "black young kids"}))

