import pandas as pd

# Carregue os dados de avaliações de usuários
ratings = pd.read_csv('ratings.csv')

# Carregue os dados de informações de filmes
movies = pd.read_csv('movies.csv')

# print(movies.head()["title"])

def extrair_ano_titulo(titulo):
    ano = titulo[-5:-1]
    return int(ano) if ano.isdigit() else 1900

# Adicione uma coluna de ano de lançamento aos dados dos filmes
movies['ano'] = movies['title'].apply(extrair_ano_titulo)

# Divida o título do filme para remover o ano (para melhor correspondência)
movies['titulo_sem_ano'] = movies['title'].str[:-7]

# Combine as avaliações de usuários com as informações dos filmes
dados = pd.merge(ratings, movies, left_on='movieId', right_on='movieId')

# Selecione as colunas relevantes para a recomendação
dados = dados[['userId', 'movieId', 'rating', 'titulo_sem_ano']]

pivo = dados.pivot_table(index='userId', columns='titulo_sem_ano', values='rating')

pivo = pivo.fillna(0)

# print(pivo)

# print(pivo.loc[1, 'Toy Story']) # Acessando o valor por linha e coluna
# print(pivo['Toy Story']) # Acecssando a coluna inteira
# print(pivo.loc[1]) # Acessando os valores de uma unica linha

from sklearn.neighbors import KNeighborsRegressor

# Crie um modelo k-NN de regressão (ajuste o número de vizinhos conforme necessário)
knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')

y = pivo.loc[1]
x = pivo.reset_index(drop=True)
