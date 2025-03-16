from transformers import pipeline

summarization = pipeline("summarization")

text = "I am boi ben, i won't back down"
summarization_text = summarization(text)

print("Summarization Text:")
print(summarization_text[0])