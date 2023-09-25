import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Crie um DataFrame de exemplo com informações de filmes e avaliações de usuários
data = {
    'Filme': ['Filme A', 'Filme B', 'Filme C', 'Filme D', 'Filme E'],
    'AvaliacaoUsuario1': [4.5, 3.0, 5.0, 2.5, 4.0],
    'AvaliacaoUsuario2': [3.5, 2.0, 4.5, 2.0, 3.5],
    'AvaliacaoUsuario3': [4.0, 2.5, 4.0, 1.5, 3.0],
}

df = pd.DataFrame(data)

# Defina o filme-alvo para o qual você deseja fazer a recomendação
filme_alvo = 'Filme D'

# Remova o filme-alvo do DataFrame para evitar recomendações para ele mesmo
df = df[df['Filme'] != filme_alvo]

# Selecione as avaliações dos usuários como características (X) e o filme-alvo como destino (y)
X = df[['AvaliacaoUsuario1', 'AvaliacaoUsuario2', 'AvaliacaoUsuario3']]
y = df[filme_alvo]

# Crie um modelo de regressão k-NN (ajuste o número de vizinhos conforme necessário)
knn_regressor = KNeighborsRegressor(n_neighbors=2)

# Treine o modelo com as avaliações dos usuários
knn_regressor.fit(X, y)

# Preveja a avaliação do filme-alvo com base nas avaliações dos usuários
avaliacao_prevista = knn_regressor.predict([[3.0, 2.0, 3.5]])

# Imprima a avaliação prevista para o filme-alvo
print(f"Avaliação prevista para {filme_alvo}: {avaliacao_prevista[0]:.2f}")