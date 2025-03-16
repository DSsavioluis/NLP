import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Carregar modelo treinado e tokenizer
model = AutoModelForSequenceClassification.from_pretrained("my_finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("my_finetuned_model")

tokenized_datasets = torch.load("preprocessed_data/dataset.pt")

# Predições e avaliação
from transformers import Trainer
trainer = Trainer(model=model)
preds_output = trainer.predict(tokenized_datasets["test"])
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = tokenized_datasets["test"]["label"]

# Gerar relatório de classificação
report = classification_report(y_true, y_pred, digits=4)
print("Classification Report\n", report)

# Gerar matriz de confusão
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.show()