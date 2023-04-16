# Movie-Recommendation-System-using-ML

This repository contains the code for a simple movie recommendation system that uses count vectorization and cosine similarities. The system recommends movies to users based on their past movie ratings.

```python

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("/content/movies (2).csv")
df.head()
df.info()
df.isnull().sum()
df.drop("homepage", axis='columns')
df.duplicated().sum()

features = ['keywords','cast','tagline','genres','director']
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['tagline']+" "+row['director']
    
    for feature in features:
    df[feature] = df[feature].fillna('')
    
    
df["combined_features"] = df.apply(combine_features,axis=1)
count = CountVectorizer()
cmatrix = count.fit_transform(df["combined_features"])
print(cmatrix)

cos_sim = cosine_similarity(cmatrix)
print(cos_sim)
print(cos_sim.shape)

movie_user_likes = input(' Enter the movie name that you recently watched: ')
all_titles = df['title'].tolist()

import difflib
Nearest_matches = difflib.get_close_matches(movie_user_likes, all_titles)
print(Nearest_matches)

close_match = Nearest_matches[0]
print(close_match)

def get_index_from_title(title):
    try:
        return df[df.title == title]["index"].values[0]
    except IndexError:
        print(f"Movie '{title}' not found in DataFrame!")
        return None

indx = get_index_from_title(movie_user_likes)

print(indx)

sim_movies = list(enumerate(cos_sim[indx]))
print(sim_movies)
print(len(sim_movies))

sorted_sim_movies = sorted(sim_movies,key=lambda x:x[1],reverse=True)[1:]
print(sorted_sim_movies)

print('Suggested movies for you : \n')
i = 1
for movie in sorted_sim_movies:
  index = movie[0]
  title_from_index = df[df.index==index]['title'].values[0]
  
  if (i<=10):
    print(i, '.',title_from_index)
    i+=1

```
