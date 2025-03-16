from transformers import pipeline

translation = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")

text = "I am boy ben, i won't back down"
translation_text = translation(text)

print("translation Text:")
print(translation_text[0])