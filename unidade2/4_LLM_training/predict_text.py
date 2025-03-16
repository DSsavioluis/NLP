from transformer_Encoder import preprocess_text
import torch
from Train_text_classifier import TransformerClassifier
import json

with open("vocab.json", "r") as f:
    vocab = json.load(f)

size_v = len(vocab)


# Função para realizar a predição de sentimento
def predict_sentiment(text, model, vocab, max_len, device):
    model.eval()
    # Pré-processa o texto e adiciona uma dimensão para o batch
    input_tensor = preprocess_text(text, vocab, max_len).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)  # outputs terá shape (1, num_classes)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class


# Lista de textos para predição
test_texts = [
    "produto excelente com ótima qualidade",
    "pior produto que já comprei",
    "não gostei do produto, qualidade ruim",
    "chegou no prazo e funciona perfeitamente",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(vocab_size=size_v, d_model=128, nhead=4,
                              num_encoder_layers=2, dim_feedforward=512,
                              num_classes=2, max_len=40, dropout=0.1)
model = model.to(device)
model.load_state_dict(torch.load("transformer_model_classification.pth", map_location=device))
model.eval()

# Realizando as predições
for text in test_texts:
    pred = predict_sentiment(text, model, vocab, max_len=40, device=device)
    label = "positivo" if pred == 1 else "negativo"
    print(f"Texto: {text}\nPredição: {label}\n")