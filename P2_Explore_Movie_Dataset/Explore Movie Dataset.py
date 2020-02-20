import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
movie_data = pd.read_csv('tmdb-movies.csv')
print(movie_data.shape)
movie_data.head()
movie_data.tail()
movie_data.sample()
movie_data.isnull()


x =  movie_data.isnull().sum().sum()
print(x)

movie_data = movie_data.fillna(method = 'ffill', axis = 0)

x =  movie_data.isnull().sum().sum()
print(x)

# Read data in the columns "id', 'popularity', 'budget', 'runtime', and 'vote_average'
movie_data[['id','popularity','budget','runtime','vote_average']]

# Read data from row 1-20, 48, and 49
movie_data[0:20].append(movie_data[47:49])

# Read data from line 50-60 of popularity
movie_data['popularity'][49:60]

# Read data that have popularity greater than 5
high_popularity = movie_data[movie_data['popularity']>5]
high_popularity.head()

# Read data that have popularity greater than 5 and later than year 1996
latest_high_popularity = movie_data[(movie_data['popularity']>5) & (movie_data['release_year']>1996)]
latest_high_popularity.head()

# group the data by revenue
movie_data.groupby(['release_year'])['revenue'].agg('mean')

# group the data by popularity
movie_data.groupby(['director'])['popularity'].agg('mean').sort_values(ascending=False)

# plot popularity data with a bar plot
df1 = movie_data.sort_values('popularity', ascending=False)
high_popularity = df1[:20]
base_color = sb.color_palette()[0]
sb.barplot(data = high_popularity, x = 'original_title', y = 'popularity', color = base_color)
plt.xticks(rotation = 90);
plt.xlabel('Title of Movies');
plt.ylabel('Popularity');

# Analyze the profit of the movies with respect to year
df2 = movie_data.sort_values('release_year', ascending=True)
df2['profit'] =  df2['revenue'] - df2['budget']
xbin_edges = np.arange(0.5, df2['release_year'].max()+1, 1)
xbin_centers = (xbin_edges + 1/2)[:-1]
data_xbins = pd.cut(df2['release_year'], xbin_edges, right = False, include_lowest = True)
y_means = df2['revenue'].groupby(data_xbins).mean()
plt.errorbar(x = xbin_centers, y = y_means);
plt.xlabel('Time(Year)');
plt.ylabel('Profit($)');
print( )
print('Analysis:')

## extract each director
tmp = movie_data['director'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('director') 
movie_data_split = movie_data[['original_title', 'revenue']].join(tmp)

top10_director = pd.Series(movie_data_split['director'].value_counts()[:10].index, name='director').reset_index() 

sort_data = movie_data.merge(top10_director, on='director')\
                      .sort_values(['index','revenue_adj'],ascending=[True,False])

## target data
target_data = sort_data.groupby('director').head(3) 

## plot
fig = plt.figure(figsize=(15, 6)) 
ax = sb.barplot(data=target_data, x='original_title', y='revenue_adj', hue='director', dodge=False)
plt.xticks(rotation = 90)
plt.ylabel('Revenue')
plt.xlabel('Original Title')
print('The profit of movies has an general ascending trend with progressing time.')
