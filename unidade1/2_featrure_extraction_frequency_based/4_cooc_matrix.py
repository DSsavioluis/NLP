import itertools
from scipy.sparse import lil_matrix

# Lista de frases
phrases = [
    "gato gosta de peixe",
    "peixe gosta de mar",
    "mar gosta de gato"
]

# Construção do vocabulário
main_dic = {'gato': 0, 'gosta': 1, 'de': 2, 'peixe': 3, 'mar': 4}


# Construção de corp_mat
corp_mat = []
for phrase in phrases:
    words = phrase.split()
    row = [(main_dic[word], 1) for word in words]
    corp_mat.append(row)

print(corp_mat)

trial = {}
for row in corp_mat:
    idx = [q[0] for q in row]
    combos = list(itertools.combinations(idx, 2))
    for tup in combos:
        if tup in trial:
            trial[tup] += 1
        else:
            trial[tup] = 1

scorp_mat = lil_matrix((len(main_dic), len(main_dic)))
for key, val in trial.items():
    scorp_mat[key[0], key[1]] = val

print("Vocabulário:", main_dic)
print("\nMatriz de Coocorrência:")
print(scorp_mat.toarray())