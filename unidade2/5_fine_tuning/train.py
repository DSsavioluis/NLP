import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# Carregar dataset pré-processado
tokenized_datasets = torch.load("preprocessed_data/dataset.pt")

# Configurar modelo
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Definir argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_steps=10,
    logging_dir="./logs"
)

# Criar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

# Executar treinamento
trainer.train()

# Avaliação do modelo
eval_results = trainer.evaluate()
print("Evaluation results: ", eval_results)

# Salvar modelo treinado
trainer.save_model("my_finetuned_model")