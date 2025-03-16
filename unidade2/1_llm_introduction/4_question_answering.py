from transformers import pipeline

# Load the pipeline for question answering
bot = pipeline("question-answering")

context = """
I am a chatbot designed to assist you with various tasks. My primary goal is to provide accurate
and helpful information. Feel free to ask me anything!
"""
question = "You are chatbot?"

# Pass the question and context to the model
question_text = bot(question=question, context=context)

print("Question answered:")
print(question_text)