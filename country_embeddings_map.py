import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Load the pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

list_of_countries = [
    "United_States", "Canada", "Germany", "France", "England", "Italy", "Spain", "China", "Japan", "India", "Brazil",
    "Russia", "Australia", "Mexico", "South_Africa", "Argentina", "Saudi_Arabia", "North_Korea", "South_Korea", "Turkey",
    "Netherlands", "Switzerland", "Sweden", "Belgium", "Norway", "Poland", "Austria", "Denmark", "Finland",
    "Greece", "Portugal", "Ireland", "New_Zealand", "Israel", "Singapore", "Malaysia", "Thailand", "Indonesia",
    "Philippines", "Vietnam", "Pakistan", "Bangladesh", "Nigeria", "Egypt", "Kenya", "Morocco", "Algeria",
    "Chile", "Colombia", "Peru", "Venezuela", "Cuba", "Hell"
]

# Get the embeddings for the list of countries
embeddings = []
missing_countries = []

for country in list_of_countries:
    if country in model:
        embeddings.append(model[country])
    else:
        missing_countries.append(country)

# Convert embeddings to a numpy array
embeddings = np.array(embeddings)

# Print countries not found in the model
if missing_countries:
    print("Countries not found in the model:", missing_countries)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot the embeddings
plt.figure(figsize=(14, 10))
texts = []
for i, country in enumerate(list_of_countries):
    if country in model:
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        texts.append(plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], country.replace('_', ' ')))

# Adjust text to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

plt.title("2D Visualization of Country Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()