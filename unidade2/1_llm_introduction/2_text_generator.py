from transformers import pipeline

generator = pipeline("text-generation", model="openai-community/gpt2")

prompt = "Once upon a time in a small village"
generated_text = generator(prompt, max_length=100, num_return_sequences=1)

# Print the generated text
print("Generated Text:")
print(generated_text[0]['generated_text'])