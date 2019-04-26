# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:26:09 2019

@author: 54329
"""

# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis


# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)



# Importing dataset
customers_df = pd.read_excel('FINALExam_Mobile_App_Survey_Data.xlsx')



########################
# Step 1: Remove demographic information
########################

customer_features_reduced = customers_df.iloc[ : , 2:-11 ]



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)



########################
# Step 3: Run PCA without limiting the number of components
########################

customer_pca_reduced = PCA(n_components = None,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)



########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 10))

features = range(customer_pca_reduced.n_components_)


plt.plot(features,
         customer_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Mobile App Research Data')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()




########################
# Step 5: Run PCA again based on the desired number of components
########################

customer_pca_reduced = PCA(n_components = 6,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)



########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(customer_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(customer_features_reduced.columns[:])


print(factor_loadings_df)


factor_loadings_df.to_excel('final_practice_factor_loadings.xlsx')



########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)


X_pca_df = pd.DataFrame(X_pca_reduced)



########################
# Step 8: Rename your principal components and reattach demographic information
########################

X_pca_df.columns = ['Early Tech Adopters', 'Introverted Geeky Learders', 'Mordern Entertainment Seekers', 'Lazy Followers', 'Brand Conscious', 'Shopoholic Musicians']


final_pca_df = pd.concat([customers_df.iloc[ : , 1:2], customers_df.iloc[ : , -11 :] , X_pca_df], axis = 1)




########################
# Step 9: Analyze in more detail
########################


# Renaming columns 
q1_values = {1: 'Under 18',
            2: '18-24',
            3: '25-29',
            4: '30-34',
            5: '35-39',
            6: '40-44',
            7: '45-49',
            8: '50-54',
            9: '55-59',
            10:'60-64',
            11:'65 or over'}

final_pca_df['q1'].replace(q1_values, inplace = True)

q48_values = {1: 'Sme High Sch',
             2: 'High Sch Grad',
             3: 'Sme Clg',
             4: 'Clg Grad',
             5: 'Sme PG Studies',
             6: 'PG Deg'}

final_pca_df['q48'].replace(q48_values, inplace = True)

## RENAMING Q49

q49_values = {1: 'Married',
             2: 'Single',
             3: 'Single with a partner',
             4: 'Separated/Widowed/Divorced'}

final_pca_df['q49'].replace(q49_values, inplace = True)

## RENAMING Q54

q54_values = {1: 'White or Caucasian',
             2: 'Black or African American',
             3: 'Asian',
             4: 'Native Hawaiian or Other Pacific Islander',
             5: 'American Indian or Alaska Native',
             6: 'Other Race'}

final_pca_df['q54'].replace(q54_values, inplace = True)

## RENAMING Q55

q55_values = {1: 'Yes',
             2: 'No'}

final_pca_df['q55'].replace(q55_values, inplace = True)

## RENAMING Q56

q56_values = {1:'Under $10,000',
            2: '$10K - $14,999',
            3: '$15K - $19,999',
            4: '$20K - $29,999',
            5: '$30K - $39,999',
            6: '$40K - $49,999',
            7: '$50K - $59,999',
            8: '$60K - $69,999',
            9: '$70K - $79,999',
            10:'$80K - $89,999',
            11:'$90K - $99,999',
            12:'$100K - $124,999',
            13:'$125K - $149,999',
            14:'$150K and over'}

final_pca_df['q56'].replace(q56_values, inplace = True)

## RENAMING Q57

q57_values = {1: 'Male',
             2: 'Female'}

final_pca_df['q57'].replace(q57_values, inplace = True)

########## Analyzing by demograhics with Shopoholic Musicians
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# Analyzing by demograhics with Mordern Entertainment Seekers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




###############################################################################
#KMeans Cluster Analysis 
###############################################################################

from sklearn.cluster import KMeans # k-means clustering

ks = range(1, 10)
inertias = []


for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)


    # Fit model to samples
    model.fit(X_scaled_reduced)


    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)



# Plot ks vs inertias
fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(ks, inertias, '-o')


plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)


plt.show()
########################
# Step 1: Remove demographic information
########################

customer_features_reduced = customers_df.iloc[ : , 2:-11 ]


########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k = KMeans(n_clusters = 3,
                      random_state = 508)


customers_k.fit(X_scaled_reduced)


customers_kmeans_clusters = pd.DataFrame({'cluster': customers_k.labels_})


print(customers_kmeans_clusters.iloc[: , 0].value_counts())



########################
# Step 4: Analyze cluster centers
########################

centroids = customers_k.cluster_centers_


centroids_df = pd.DataFrame(centroids)



# Renaming columns
centroids_df.columns = customer_features_reduced.columns


print(centroids_df)


# Sending data to Excel
centroids_df.to_excel('Finalexam_customers_k3_centriods.xlsx')



########################
# Step 5: Analyze cluster memberships
########################


X_scaled_reduced_df = pd.DataFrame(X_scaled_reduced)


X_scaled_reduced_df.columns = customer_features_reduced.columns


clusters_df = pd.concat([customers_kmeans_clusters,
                         X_scaled_reduced_df],
                         axis = 1)


print(clusters_df)



########################
# Step 6: Reattach demographic information 
########################
final_clusters_df = pd.concat([customers_df.iloc[ : , 1:2], customers_df.iloc[ : , -11 :] ,clusters_df ], axis = 1)



print(final_clusters_df)

final_clusters_df.to_excel('finalexam_customers_cluster_centriods2.xlsx')



########################
# Step 7: Analyze in more detail 
########################


# Renaming demographic columns 
q1_values = {1: 'Under 18',
            2: '18-24',
            3: '25-29',
            4: '30-34',
            5: '35-39',
            6: '40-44',
            7: '45-49',
            8: '50-54',
            9: '55-59',
            10:'60-64',
            11:'65 or over'}

final_pca_df['q1'].replace(q1_values, inplace = True)

q48_values = {1: 'Sme High Sch',
             2: 'High Sch Grad',
             3: 'Sme Clg',
             4: 'Clg Grad',
             5: 'Sme PG Studies',
             6: 'PG Deg'}

final_pca_df['q48'].replace(q48_values, inplace = True)


q49_values = {1: 'Married',
             2: 'Single',
             3: 'Single with a partner',
             4: 'Separated/Widowed/Divorced'}

final_pca_df['q49'].replace(q49_values, inplace = True)


q54_values = {1: 'White or Caucasian',
             2: 'Black or African American',
             3: 'Asian',
             4: 'Native Hawaiian or Other Pacific Islander',
             5: 'American Indian or Alaska Native',
             6: 'Other Race'}

final_pca_df['q54'].replace(q54_values, inplace = True)


q55_values = {1: 'Yes',
             2: 'No'}

final_pca_df['q55'].replace(q55_values, inplace = True)


q56_values = {1:'Under $10,000',
            2: '$10K - $14,999',
            3: '$15K - $19,999',
            4: '$20K - $29,999',
            5: '$30K - $39,999',
            6: '$40K - $49,999',
            7: '$50K - $59,999',
            8: '$60K - $69,999',
            9: '$70K - $79,999',
            10:'$80K - $89,999',
            11:'$90K - $99,999',
            12:'$100K - $124,999',
            13:'$125K - $149,999',
            14:'$150K and over'}

final_pca_df['q56'].replace(q56_values, inplace = True)


q57_values = {1: 'Male',
             2: 'Female'}

final_pca_df['q57'].replace(q57_values, inplace = True)


########## Analyzing by demograhics with Shopoholic Musicians
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# Analyzing by demograhics with Mordern Entertainment Seekers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




###############################################################################
# Combining PCA and Clustering!!!
###############################################################################


########################
# Step 1: Take your transformed dataframe
########################

print(X_pca_df.head(n = 5))


print(pd.np.var(X_pca_df))



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)


customers_k_pca.fit(X_pca_clust_df)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})


print(customers_kmeans_pca.iloc[: , 0].value_counts())




########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['Early Tech Adopters', 'Introverted Geeky Learders', 'Mordern Entertainment Seekers', 'Lazy Followers', 'Brand Conscious', 'Shopoholic Musicians']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('Finalexam_customers_pca_centriods.xlsx')



########################
# Step 5: Analyze cluster memberships
########################

clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)



########################
# Step 6: Reattach demographic information
########################

final_pca_clust_df = pd.concat([customers_df.iloc[ : , 1:2], customers_df.iloc[ : , -11 :],
                                clst_pca_df],
                                axis = 1)

final_clusters_df = pd.concat([customers_df.iloc[ : , 1:2], customers_df.iloc[ : , -11 :] ,clusters_df ], axis = 1)



print(final_pca_clust_df.head(n = 5))



########################
# Step 7: Analyze in more detail 
########################

# Renaming demographic columns 
q1_values = {1: 'Under 18',
            2: '18-24',
            3: '25-29',
            4: '30-34',
            5: '35-39',
            6: '40-44',
            7: '45-49',
            8: '50-54',
            9: '55-59',
            10:'60-64',
            11:'65 or over'}

final_pca_df['q1'].replace(q1_values, inplace = True)

q48_values = {1: 'Sme High Sch',
             2: 'High Sch Grad',
             3: 'Sme Clg',
             4: 'Clg Grad',
             5: 'Sme PG Studies',
             6: 'PG Deg'}

final_pca_df['q48'].replace(q48_values, inplace = True)


q49_values = {1: 'Married',
             2: 'Single',
             3: 'Single with a partner',
             4: 'Separated/Widowed/Divorced'}

final_pca_df['q49'].replace(q49_values, inplace = True)


q54_values = {1: 'White or Caucasian',
             2: 'Black or African American',
             3: 'Asian',
             4: 'Native Hawaiian or Other Pacific Islander',
             5: 'American Indian or Alaska Native',
             6: 'Other Race'}

final_pca_df['q54'].replace(q54_values, inplace = True)


q55_values = {1: 'Yes',
             2: 'No'}

final_pca_df['q55'].replace(q55_values, inplace = True)


q56_values = {1:'Under $10,000',
            2: '$10K - $14,999',
            3: '$15K - $19,999',
            4: '$20K - $29,999',
            5: '$30K - $39,999',
            6: '$40K - $49,999',
            7: '$50K - $59,999',
            8: '$60K - $69,999',
            9: '$70K - $79,999',
            10:'$80K - $89,999',
            11:'$90K - $99,999',
            12:'$100K - $124,999',
            13:'$125K - $149,999',
            14:'$150K and over'}

final_pca_df['q56'].replace(q56_values, inplace = True)


q57_values = {1: 'Male',
             2: 'Female'}

final_pca_df['q57'].replace(q57_values, inplace = True)

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Shopoholic Musicians',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



# Analyzing by demograhics with Mordern Entertainment Seekers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Mordern Entertainment Seekers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()