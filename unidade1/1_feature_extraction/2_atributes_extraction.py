# Atividade

# 1) Um função que retorne a quantidade de sentenças em um texto.

import re
import pandas as pd

def qnt_sentencas(texto):
    sentenca_padrao = r'[.!?]'
    return len(re.findall(sentenca_padrao,texto))

texto1 = 'cuida na fuga de irauçuba. cuida cuida.'

contagem = qnt_sentencas(texto1)

print(contagem)


# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.

def qnt_maiousculas(texto):
    sentenca_padrao = r'\b[A-Z][a-z]+\b'
    return len(re.findall(sentenca_padrao,texto))

texto2 = 'Cuida na Fuga de Iraucuba'

contagemm = qnt_sentencas(texto2)

print(contagemm)

# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.

def qnt_numbers(texto):
    sentenca_padrao = r'\d[0-9]+'
    return len(re.findall(sentenca_padrao,texto))

texto3 = 'cuida na Fuga de irauçuba 33, 55646, 95425'

contagemm = qnt_sentencas(texto3)

print(contagemm)

# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.

def qnt_caixa_alta(texto):
    sentenca_padrao = r'\w[A-Z]+'
    return len(re.findall(sentenca_padrao,texto))

texto4 = 'cuida na FUGA de irauçuba!'

contagemm = qnt_sentencas(texto4)

print(contagemm)

# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.

df = pd.DataFrame({"text": [texto1, texto2, texto3, texto4]})
print(df)

# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.
df_resultados = pd.DataFrame({
    "Sentenças": df["text"].apply(qnt_sentencas),
    "Palavras com maiúscula": df["text"].apply(qnt_maiousculas),
    "Números": df["text"].apply(qnt_numbers),
    "Palavras em caixa alta": df["text"].apply(qnt_caixa_alta)
})

print(df_resultados)