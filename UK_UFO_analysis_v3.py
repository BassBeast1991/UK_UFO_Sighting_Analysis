# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 12:30:11 2023

@author: aidan
"""

#______________________________________________________________________________
# Import libraries
import camelot 
import pandas as pd
import regex as re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk.corpus
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
import itertools
import collections
from nltk import bigrams
import networkx as nx
import os
import numpy as np

#______________________________________________________________________________
# Read the file
datadir = '.\\'

filenames = os.listdir(datadir)

initial = True

for file in filenames:
    
    print('Current file is ' + file)
    tables = camelot.read_pdf(datadir+file,pages='all')
    
#______________________________________________________________________________
# Format the data
    
    # get column names
    if initial == True:
        col_names = ['Date','Time','Town','County','Occupation','Description'] #tables[0].df.iloc[0]
    tables[0].df = tables[0].df.iloc[1:]
    
    # change column names
    for n in range(0,len(tables)):
        
        temp_df = tables[n].df
        temp_df.columns = col_names
        temp_df.drop(temp_df[temp_df['Description'].str.len() < 50].index, inplace=True)
        
        if (n==0 and initial == True):
            
            df = temp_df
            initial = False
            
        else:
            
            df = pd.concat([df,temp_df])

#______________________________________________________________________________
# clean the data

# drop duplicates
df.drop_duplicates(subset='Description',inplace=True)

# convert timestamps to a common datetime format
import dateparser
import datetime

print('Processing sighting dates...')

# desired common date format
dmy_re = re.compile(r"^(?P<day>\d+)/(?P<month>\d+)/(?P<year>\d+)$")

# function to process each sighting date
def parse_date(ds, regexps=()):
    for regexp in regexps:
        match = regexp.match(ds)
        if match:
            return datetime.datetime(**{k: int(v) for (k, v) in match.groupdict().items()})
    return dateparser.parse(ds)
        
# create set of datetimes using function with a list comprehension
datetimes = [parse_date(d, regexps=[dmy_re]) for d in df['Date']]
        
# get rid of any null values from both datetimes list and dataset
none_inds = [i for i in range(len(datetimes)) if (datetimes[i] == None)]
df.reset_index(drop=True,inplace=True)
df.drop(none_inds,inplace=True)
        
for idx in sorted(none_inds, reverse=True):
    del datetimes[idx]
        
df['Date'] = pd.Series(datetimes)
        
#should be no Nulls
num_nans = pd.isnull(df).sum()
print(num_nans)
    
# delete nulls
df.dropna(how='any',inplace=True)

# replace original data indexing with datetimes
df.set_index('Date',inplace=True,drop = True)

# print data summary...
print('Cleaning text...')

# apply same cleaning process to each description in dataframe
stop = stopwords.words('english')
stemmer = SnowballStemmer("english")

df_original = df.copy()

print(df_original.head)
print(df_original.tail)

df['Description']=df['Description'].apply(lambda x : x.lower())
df['Description']=df['Description'].str.replace(r'[0-9]+', '', regex=True)
df['Description']=df['Description'].str.replace(r'[^\w\s]', '', regex=True)
df['Description']=df['Description'].str.replace('\n', '', regex=False)
df['Description']=df['Description'].apply(lambda x : ' '.join([word for word in x.split(' ') if (word not in (stop) and len(word)>3)]))
#df['Description']=df['Description'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

# remove most common words from cleaned text
most_common_all = list(collections.Counter(' '.join(df['Description']).split(' ')).
                       most_common(30))

most_common_all = [x[0] for x in most_common_all]

expr_regx = r'({})'.format('|'.join(most_common_all))

df['Description'] = df['Description'].str.replace(expr_regx, '', regex=True)
 
df['Description'] = df['Description'].apply(lambda x : ' '.join([word for word in x.split(' ') if len(word)>1]))


# clean the description text as list for word cloud and bigrams
cleaned_text = ' '.join(df['Description'])

#______________________________________________________________________________
# Create word cloud from the descriptions

print('Creating word cloud...')

UFO_description_cloud = WordCloud(width=800, height=400,collocations = False, 
            background_color = 'white').generate(cleaned_text)

plt.imshow(UFO_description_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

UFO_description_cloud.to_file('UFO_Word_Cloud.png')

#______________________________________________________________________________
# Create bigram network from the descriptions

print('Creating bigram network...')

# list of words in each Tweet
words_in_text = cleaned_text.split(' ')

# get list of bigrams (words which occur next to each other) from the Tweets
bigram_lists = list(bigrams(words_in_text))

# convert bigrams from 'list of lists' to a simple list
all_bigrams = list(itertools.chain(*bigram_lists))

# Create counter of words in clean bigrams
bigram_occurences = collections.Counter(bigram_lists)

top_bigram_occurences = bigram_occurences.most_common(200)

# convert to a dataframe for display as word network
bigram_df = pd.DataFrame(top_bigram_occurences,
                             columns=['Bigram', 'Number of Occurences'])

# generate dictionary for network plot
top_bigram_occurences_dict = bigram_df.set_index('Bigram').T.to_dict('records')

# generate the network  
graph_of_tweet_bigrams = nx.Graph()

# connect the nodes (i.e. words)
for a, b in top_bigram_occurences_dict[0].items():
    graph_of_tweet_bigrams.add_edge(a[0], a[1], weight=(b * 10))

# Plot the network of bigrams
fig, ax = plt.subplots(figsize=(25, 20))

# position of each node, for adding labels
node_positions = nx.spring_layout(graph_of_tweet_bigrams, k=2)

# Plot networks
nx.draw_networkx(graph_of_tweet_bigrams, node_positions,
                 font_size=20,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels = False,
                 ax=ax)


# Create offset labels to show the words
for key, value in node_positions.items():
    x, y = value[0]+.1, value[1]+.02
    
    this_label = key
    
    ax.text(x, y,
            s=this_label,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=11)

fig.savefig("UFO_Bigrams.png")

#______________________________________________________________________________
# Tokenize data using word embedding and reduce dimensionality using PCA

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min, pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram

# tokenize wordds using embedding
tokenizer = SentenceTransformer('bert-base-nli-mean-tokens')

tokens = [tokenizer.encode(df['Description'].iloc[i]) for i in range(len(df))]

token_df = pd.DataFrame(tokens,columns = ["Component "+str(i) for i in range(len(tokens[0]))],index=df.index)

scaler = StandardScaler()
scaler.fit(token_df)
token_array = scaler.transform(token_df)

# get rid of null values
num_nans = pd.isnull(token_df).sum().sum()
print(num_nans)

#______________________________________________________________________________
# Dimensionality reduction using PCA

pca_data = PCA(n_components=2).fit_transform(token_array)

#______________________________________________________________________________
# Apply agglomerative clustering, using silouhette score to determine the 
# number of clusters


n_clusters = list(range(2,20))
silhouette_avg_array = []

for ind in range(len(n_clusters)):

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    agglom = AgglomerativeClustering(n_clusters = n_clusters[ind])
    cluster_labels = agglom.fit_predict(pca_data)
    cluster_labels = agglom.labels_
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(pca_data, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters[ind],
        "The average silhouette_score is :",
        silhouette_avg)
    silhouette_avg_array.append(silhouette_avg)

n_clusters_idx = silhouette_avg_array.index(min(silhouette_avg_array))
n_clusters_opt = n_clusters[n_clusters_idx]

print(str(n_clusters_opt) + 'clusters were detected.')

agglom = AgglomerativeClustering(n_clusters = n_clusters_opt,compute_distances=True)

#agglom = AgglomerativeClustering(n_clusters = None, distance_threshold=200,compute_full_tree=True)
agglom.fit(pca_data)

#______________________________________________________________________________
# plot clustered data

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.5  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = pca_data[:, 0].min() - 1, pca_data[:, 0].max() + 1
y_min, y_max = pca_data[:, 1].min() - 1, pca_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z_pred = agglom.fit_predict(np.c_[xx.ravel(), yy.ravel()])
Z_labels = agglom.labels_

# Put the result into a color plot
Z_pred = Z_pred.reshape(xx.shape)

# predictions for the actual data
Z = agglom.fit_predict(pca_data)
Z_labels = agglom.labels_

# Put the result into a color plot
#Z = Z.reshape(xx.shape)
fig, ax = plt.subplots(figsize=(10, 8))


plt.imshow(
    Z_pred,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)


plt.plot(pca_data[:, 0], pca_data[:, 1], "k.", markersize=10)

nc = NearestCentroid()
centroids = nc.fit(pca_data, Z)
centroids = centroids.centroids_

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=300,
    linewidths=3,
    color="white",
    zorder=10,
)
plt.title(
    "Agglomerative Clustering of UFO Sightings\n"
    "PCA-Reduced Dimensionality, Centroids = Black Cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

fig.savefig("UFO_PCA_Clustering.png")

#______________________________________________________________________________
# Extract closest description to each centroid and separate data out by cluster
closest_description, _ = pairwise_distances_argmin_min(centroids, pca_data)

# get index of closest centroid to each description
closest_centroid, _ = pairwise_distances_argmin_min(pca_data,centroids)

typical_descriptions = df_original['Description'].iloc[closest_description]

#______________________________________________________________________________
# furthest datapoints from all centroids

distances = pairwise_distances(pca_data,centroids).sum(axis=1)
dist_sort_index = np.argsort(distances)
dist_sort_index = dist_sort_index[-10:]

furthest_descriptions = df_original['Description'].iloc[dist_sort_index]

for k in range(len(furthest_descriptions)):
    
    print(str(k+1) + ' - ' + furthest_descriptions[k])
    print('\n')

#______________________________________________________________________________
# word cloud for each class of sighting

df_classes = list()
df_classes_clean = list()

df.to_csv('All_UFO_Classes.csv')

word_count = list()
most_common_per_class = list()

for k in range(len(closest_description)):
    
    print(str(k+1) + ' - ' + typical_descriptions[k])
    print('\n')

    these_df_inds = df.index[np.where(closest_centroid == k)]
    
    # get indices from distance to centroid
    this_class_df = df_original.loc[these_df_inds]
    this_class_df.drop_duplicates(subset='Description',inplace=True)
    
    this_class_df_clean = df.loc[these_df_inds]
    this_class_df_clean.drop_duplicates(subset='Description',inplace=True)
    
    df_classes.append(this_class_df)
    df_classes_clean.append(this_class_df_clean)
    
    this_class_df.to_csv('UFO_Class_'+str(k+1)+'.csv')
    
    desc_strs = ' '.join(this_class_df_clean['Description'])
    desc_strs = desc_strs.replace('\n','')
    
    # find most common words in this class dataframe
    word_count.append(collections.Counter(desc_strs.split(' ')).most_common(30))
    most_common_per_class.append(list(map(lambda x: x[0], word_count[k])))
    
# flatten most common words into simple list and get intersection
most_common_per_class = list(set.intersection(*list(map(set, most_common_per_class))))

# create word cloud for each cluster
for k in range(len(closest_description)):    
    
   expr_regx = r'({})'.format('|'.join(most_common_per_class))
   df_classes_clean[k]['Description'] = df_classes_clean[k]['Description'].str.replace(expr_regx, '', regex=True)
    
   df_classes_clean[k]['Description'] = df_classes_clean[k]['Description'].apply(lambda x : ' '.join([word for word in x.split(' ') if len(word)>1]))
   
   UFO_description_cloud = WordCloud(width=800, height=400,collocations = False, 
                background_color = 'white').generate(' '.join(df_classes_clean[k]['Description']))

   plt.imshow(UFO_description_cloud, interpolation='bilinear')
   plt.axis("off")
   plt.show()

   UFO_description_cloud.to_file('UFO_Word_Cloud_' +str(k+1)+ '.png')
    
# plot dendrogram for clustering
def plot_dendrogram(model, **kwargs):

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plt.figure(figsize=(10,8))
plt.title("Hierarchical Clustering Dendrogram")

# plot the top three levels of the dendrogram
plot_dendrogram(agglom.fit(pca_data), truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
plt.savefig("UFO_Dendrogram.png")




