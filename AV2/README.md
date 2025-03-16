# AVALIAÇÃO 2
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

EMILY CAMELO MENDONCA: https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets

ERICK RAMOS COUTINHO: https://www.kaggle.com/datasets/hrmello/brazilian-portuguese-hatespeech-dataset

ERYKA CARVALHO DA SILVA: https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets?select=buscape.csv

LUCIANA SOUSA MARTINS: https://www.kaggle.com/datasets/moesiof/portuguese-narrative-essays

LUIS SAVIO GOMES ROSA : https://github.com/kamplus/FakeNewsSetGen/tree/master/Dataset

MAVERICK ALEKYNE DE SOUSA RIBEIRO: https://www.kaggle.com/datasets/brunoluvizotto/brazilian-headlines-sentiments

PAULO HENRIQUE SANTOS MARQUES: https://huggingface.co/datasets/nilc-nlp/assin

SHELDA DE SOUZA RAMOS: https://huggingface.co/datasets/johnidouglas/twitter-sentiment-pt-BR-md-2-l


### Questão 1

[encoder.py](encoder.py)

Nesta questão você deve aplicar o encoder no seu dataset. Utilize os dados textuais e
realize o proceso de encoder, faça a implementação manual.

### Questão 2

[preprocessing.py](preprocessing.py)

nessa questão você deve implementar a preparação do dos dados para serem utilizados em treinamento de LLM
via pytorch. Neste etapa você deve ser capaz de ter os dados de treino e teste no formato adequado que serão utilizados nos modelos nos modelos posteriormente.
Lembre-se que todos os modelos devem treinar e testar com os mesmos conjuntos para garantir uma comparação justa.

### Questão 3

[hugging_face_models.py](hugging_face_models.py)

Utilizando os modelos pre-treinados que estão na huggingFace, sem realizar nenhum fine tuning,
faça a prediçao no conjunto de teste obtido na questão 2. Gere o classification report
e a matriz confusão. 

### Questão 4

[fine_tuning.py](fine_tuning.py)

Nesta questão você deve gerar um modelo via fine tuning, escolha o modelo apropridado como base
e especifique para sua aplicação. Treine com o conjunto de treino resposta da questão 2.


### Questão 5

[train_classifier_encoder.py](train_classifier_encoder.py)

Nesta questão você deve criar a sua arquitetura de encoder para classificar
as informações textuais. Treine com o conjunto de treino utilizado na questão 2.

### Questão 6

[prediction.py](prediction.py)

Nesta questão você deve realizar as predições dos modelos treinados nas questões 5 e 4. Faça um clasification report
e matriz confusão para cada modelo. 

### Observações para o Relatório 

No relatório, organize a comparação de resultados entre os modelos, capture as informações de 
resultados da AV1 para enriquecer a comparação. 

**IMPORTANTE: Se o seu modelo com a arquitetura própria conseguir atigir resultados próximos ou superior aos pre-treinados (se a taxa destes forem acima de 0.85),
você consegue 1 ponto extra na prova.**

Discutir **organizadamente** os resultados obtidos de cada questão-.
Ao concluir o relatório, compartilhar com **alysonbnr@ufc.br** até 18-02


### Observações para a Apresentação

Criar apresentação para realizar até 18-02