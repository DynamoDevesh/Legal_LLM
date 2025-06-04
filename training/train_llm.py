from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Load dataset
dataset = load_dataset("json", data_files="/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/data/legal_qa_sft.json")["train"]

# Load base model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize dataset
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(preprocess)

# Define training config
sft_config = SFTConfig(
    output_dir="./legal-flan-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    fp16=False,
    push_to_hub=False,
    dataset_text_field="text"
)

# Train
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)
trainer.tokenizer = tokenizer
trainer.train()
