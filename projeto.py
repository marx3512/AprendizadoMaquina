# import pandas as pd
# from scipy.sparse import csr_matrix
# from sklearn.neighbors import NearestNeighbors

# movies_filename = 'movies.csv'
# ratings_filename = 'ratings.csv'

# df_movies = pd.read_csv(
#     movies_filename,
#     usecols=['movieId', 'title'],
#     dtype={'movieId': 'int32', 'title': 'str'})

# df_ratings = pd.read_csv(
#     ratings_filename,
#     usecols=['userId', 'movieId', 'rating'],
#     dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# #print(df_movies.head())

# #print(df_ratings.head())

# df_movie_features = df_ratings.pivot(
#     index='movieId',
#     columns='userId',
#     values='rating'
# ).fillna(0)

# mat_movie_features = csr_matrix(df_movie_features.values)

# print(df_movie_features.head())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

dataFrameUser = pd.read_csv(
    'ratings1.csv',
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# dataFrameMovie = pd.read_csv(
#     'movies.csv',
#     usecols=['movieId', 'title'],
#     dtype={'movieId': 'int32', 'title': 'str'})

filmeAlvo = '3'

dataFrameUser = dataFrameUser[dataFrameUser['movieId'] != filmeAlvo]

x = dataFrameUser[['rating']]

y = dataFrameUser['movieId']

knn_regressor = KNeighborsRegressor(n_neighbors=5)

knn_regressor.fit(x, y)

avaliacao_prevista = knn_regressor.predict([[4.0]])

print(f"Avaliação prevista para {filmeAlvo}: {avaliacao_prevista[0]:.2f}")

# print(dataFrameUser.head())

# dFrameataUserTrain, dataUserTest, dataFrameMovieTrain, dataFrameMovieTest = train_test_split(dataFrameUser, dataFrameMovie, test_size=0.3, random_state=42)