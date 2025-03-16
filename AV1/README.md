# AVALIAÇÃO 1
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

[preprocessing.py](preprocessing.py)

Nesta primeira questão você deve implementar funções de manipulação do dataset realizar os pré-processmentos necessários, como stemming, lemmatização, remoção de carateris maiusculos, verificar stopwords, verificação de menções
, de acordo as características do seu dataset. Em resume prepare o mesmo para aplicação de extração de atributos. A estrutura do código deve permitir que possam ser importadas as funções em outras questões.


### Questão 2

[atribute_extraction.py](atribute_extraction.py)

Voce deve implementar funções para extração de atributos com Analise estatística individual, CountVectorizer, TF-IDF, matriz de coocorrência e word2vec.. Faça uma função para cada forma de extração de atributo, sugiro que seja construída uma classe para essas funções.
A estrutura do código deve permitir que possam ser importadas as funções em outras questões.


### Questão 3

[classification.py](classification.py)


Neste exercicio você deve utilizar um único classifiador para aplicar no seu dataset, de acordo com a label escolhida.
Voce deve comparar os resultados quantitiativos nos seguintes casos:

a) Utilizando todas as formas de extração de atributos, compare os resultados de acerto do classificador com pré-processamento e sem pré-processamento. Mostre as taxas para os casos de forma organizada.

b) Compare as formas de extração de atributos, usando com pré-processamento que você escolheu no item a)

C) Faça um variação de dois pré-procesamentos que compare lemmatização e steming, considerando a melhor forma de extração de atributos vista no item b)

 

### Observações para o Relatório 

Discutir **organizadamente** os resultados obtidos de cada questão.
Ao concluir o relatório, compartilhar com **alysonbnr@ufc.br** até 15/08


### Observações para o Apresentação

Criar apresentação para realizar até 15/08