# %%
import pandas as pd
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# %%
"""
## 1. Import data
"""

# %%
data = pd.read_excel (r'./T4.xlsx')

# %%
"""
#### Display the the number of empty values in the dataset
"""

# %%
def total_of_missing_values(data):
    missing_data = data.isna().sum().sum()
    print("\nNumber of NaN values:", missing_data)
    #return missing_data

# %%
"""
#### Display the the number and the percent of missing values for each column in the dataset
"""

# %%
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (100*data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data[missing_data['Percent']>0])

# %%
"""
#### Show the dataset shape
"""

# %%
def show_data_dimensions(data):
    print('number of rows : '+str(data.shape[0])+', number of columns : '+str(data.shape[1]))
    #return data.shape[0]

# %%
"""
#### Clean the data by dropping the duplicates rows and also by filling the empty values with previous values
"""

# %%
def preprocessing(original_data):

    clean_data = original_data.copy()

    ### drop duplicate rows
    clean_data.drop_duplicates(keep='first', inplace=True)

    ### fill missing data and drop the remaing
    clean_data = clean_data.fillna(method='ffill')
    clean_data.dropna(axis=0, inplace=True)
    return clean_data

# %%
total_of_missing_values(data)

# %%
missing_data(data)

# %%
show_data_dimensions(data)

# %%
"""
## Clean the data
"""

# %%
data2 = preprocessing(data)

# %%
#Convert all the data to string so that we can use combination of columns
data2 = data2.applymap(str)

# %%
#concate columns to use them for recommendation
data2['version_Battery_level'] = data2['Version'] +"_"+data2['Battery_level']

# %%
###This bloc is used only because we have a large datasetand the memory cannot support this.
#if you have a powerful pc you can remove this bloc
data2 = data2.iloc[:20000]
#print(len(data), len(data2))

# %%
#Here we do some processing to remove all english stop words if we have in our data set
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')
#Replace NaN with an empty string
data2['version_Battery_level'] = data2['version_Battery_level'].fillna('')
#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(data2['version_Battery_level'])
#Output the shape of tfidf_matrix
#print(tfidf_matrix.shape)

# %%
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#print(cosine_sim.shape)

# %%
#Construct a reverse map of indices and version_Battery_level
indices = pd.Series(data2.index, index=data2['version_Battery_level']).drop_duplicates()

# %%
# Function that takes in version and Battery_level as input and outputs most top 10
def get_recommendations(version,Battery_level, cosine_sim=cosine_sim):
    version_Battery_level = version+'_'+str(Battery_level)
    # Get the index of the version_Battery_level that matches the version_Battery_level
    idx = indices[version_Battery_level]
    
    # Get the pairwsie similarity scores of all version_Battery_level with that version_Battery_level
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the version_Battery_level based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: max(x[1]), reverse=True)

    # Get the scores of the 10 most similar version_Battery_level
    sim_scores = sim_scores[1:6]

    # Get the version_Battery_level indices
    version_Battery_level_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar version_Battery_level
    return data2.iloc[version_Battery_level_indices]

# %%
print(get_recommendations('2.3.1',61.0))
# %%
