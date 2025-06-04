from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/legal-flan-model/checkpoint_1000"

# ✅ Load the tokenizer and model from local files
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

def chat(prompt):
    input_text = f"Answer the legal question: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n💬", response.strip())

# 🔍 Try it
chat("What is the punishment for theft under IPC?")
