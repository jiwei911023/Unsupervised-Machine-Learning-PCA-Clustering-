# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:32:03 2019

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
final_df = pd.read_excel('finalExam_Mobile_App_Survey_Data.xlsx')


final_df.info()


final_desc = final_df.describe(percentiles = [0.01,
                                              0.05,
                                              0.10,
                                              0.25,
                                              0.50,
                                              0.75,
                                              0.90,
                                              0.95,
                                              0.99]).round(2)



final_desc.loc[['min',
                '1%',
                '5%',
                '10%',
                '25%',
                'mean',
                '50%',
                '75%',
                '90%',
                '95%',
                '99%',
                'max'], :]


# Viewing the first few rows of the data
final_df.head(n = 5)



########################
# Correlation analysis
########################


df_corr = final_df.corr().round(2)

print(df_corr)


########################
# Scaling (normalizing) variables before correlation analysis
########################


"""
REMOVE DEMOGRAPHIC INFORMATION
Demographics are characteristics of a population. Characteristics such as race,
ethnicity, gender, age, education, profession, occupation, income level, and 
marital status, are all typical examples of demographics that are used in surveys.
"""

# Scaling the wholesale customer dataset

final_features = final_df.loc[:,['q2r1','q2r2','q2r3','q2r4','q2r5','q2r6','q2r7',
                                 'q2r8','q2r9',
                                
                                 'q4r1','q4r2','q4r3','q4r4','q4r5','q4r6','q4r7',
                                 'q4r8','q4r9','q4r10','q4r11',
                                
                                 'q11',
                                 'q12',
                                
                                 'q13r1','q13r2','q13r3','q13r4','q13r5','q13r6',
                                 'q13r7','q13r8','q13r9','q13r10','q13r11','q13r12',
                                
                                 'q24r1','q24r2','q24r3','q24r4','q24r5','q24r6',
                                 'q24r7','q24r8','q24r9','q24r10','q24r11','q24r12',
                                
                                 'q26r3','q26r4','q26r5','q26r6',
                                 'q26r7','q26r8','q26r9','q26r10','q26r11','q26r12',
                                 'q26r13','q26r14','q26r15','q26r16','q26r17']]
                                
# Scaling using StandardScaler()
scaler = StandardScaler()



scaler.fit(final_features)



X_scaled = scaler.transform(final_features)



X_scaled_df = pd.DataFrame(X_scaled)



#  Checking pre- and post-scaling of the data
print(pd.np.var(final_features))
print(pd.np.var(X_scaled_df))


# Adding labels to our scaled DataFrame
X_scaled_df.columns = final_features.columns



########################
# Performing PCA on the scaled data
########################


# Looking at all of the principal components
APP_pca = PCA(n_components = None,
                   random_state = 508)



# Fitting the PCA model
APP_pca.fit(X_scaled)



# Transform data
APP_pca.transform(X_scaled)


print("Original shape:", X_scaled.shape)
print("Reduced shape:",  APP_pca.transform(X_scaled).shape)


# Explained variance as a ratio of total variance
APP_pca.explained_variance_ratio_


# Plotting the principal components
fig, ax = plt.subplots(figsize=(12, 8))

features = range(APP_pca.n_components_)


plt.bar(x = features,
        height = APP_pca.explained_variance_ratio_)


plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)

plt.show()

########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(APP_pca.n_components_)


plt.plot(features,
         APP_pca.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Wholesale Customer Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()




########################
# Step 5: Run PCA again based on the desired number of components
########################

APP_pca = PCA(n_components = 5,
                   random_state = 508)


APP_pca.fit(X_scaled)




########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(APP_pca.components_))


factor_loadings_df = factor_loadings_df.set_index(final_features.columns)

factor_loadings_df = factor_loadings_df.round(2)

print(factor_loadings_df)


factor_loadings_df.to_excel('practice_factor_loadings.xlsx')

'''
We can use this to interpreter the results, the variables that have higher 
correlation define the CPA. There are a few interesting findings here, use
CP3 and CP4. Apple users and android users.
'''

########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = APP_pca.transform(X_scaled_df)


X_pca_df = pd.DataFrame(X_pca_reduced)


########################
# Step 8: Rename your principal components and reattach demographic information
########################
'''
Rename the pca groups that the machine created
'''

X_pca_df.columns = ['A little above mean', 'Middle Age Antisocial', 
                    'Tipycal Millenial','Apple lovers','Android lovers']

final_clusters_df = pd.concat([final_df.loc[ : , ['q1',
                                                  'q48',
                                                  'q49',
                                                  
                                                  'q50r1','q50r2','q50r3',
                                                  'q50r4','q50r5',
                                                  
                                                  'q54',
                                                  'q55',
                                                  'q56',
                                                  'q57'] ],
                               
                               X_pca_df],
                               axis = 1)


print(final_clusters_df)




###############################################################################
# Cluster Analysis One More Time!!!
###############################################################################

from sklearn.cluster import KMeans # k-means clustering


########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k = KMeans(n_clusters = 8,
                      random_state = 508)


customers_k.fit(X_scaled_df)


customers_kmeans_clusters = pd.DataFrame({'cluster': customers_k.labels_})


print(customers_kmeans_clusters.iloc[: , 0].value_counts())



########################
# Step 4: Analyze cluster centers
########################

centroids = customers_k.cluster_centers_


centroids_df = pd.DataFrame(centroids)



# Renaming columns
centroids_df.columns = final_features.columns


print(centroids_df)


# Sending data to Excel
centroids_df.to_excel('customers_k3_centriods.xlsx')


###############################################################################
# Plotting Intertia
###############################################################################

"""
How many clusters do we need? Which number of clusters is 'best' for our data? 
These questions can be answered using the metric inertia.
"""


ks = range(1, 50)
inertias = []


for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)


    # Fit model to samples
    model.fit(X_scaled_df)


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
# Step 5: Analyze cluster memberships
########################


X_scaled_reduced_df = pd.DataFrame(X_scaled_df)


X_scaled_reduced_df.columns = final_features.columns


clusters_df = pd.concat([customers_kmeans_clusters,
                         X_scaled_reduced_df],
                         axis = 1)


print(clusters_df)


########################
# Step 6: Reattach demographic information 
########################

final_clusters_df = pd.concat([final_df.loc[ : , ['q1',
                                                  
                                                  'q48',
                                                  'q49',
                                                  
                                                  'q50r1','q50r2','q50r3',
                                                  'q50r4','q50r5',
                                                  
                                                  'q54',
                                                  'q55',
                                                  'q56',
                                                  'q57'] ],
                               
                               clusters_df],
                               axis = 1)


print(final_clusters_df)


########################
# Step 7: Analyze in more detail 
########################
                                    
# Renaming age
age = {1 : 'Under 18',
       2 : '18-24',
       3 : '25-29',
       4 : '30-34',
       5 : '35-39',
       6 : '40-44',
       7 : '45-49',
       8 : '50-54',
       9 : '55-59',
       10 : '60-64',
       11 : '65+'}


final_clusters_df['q1'].replace(age, inplace = True)


# Renaming Education
education = {1 : 'Some High School',
             2 : 'High School graduate',
             3 : 'Some College',
             4 : 'College Graduate',
             5 : 'Some Post-Graduate',
             6 : 'Post-Graduate Degree'}


final_clusters_df['q48'].replace(education, inplace = True)


# Renaming Marital Status
Marital = {1 : 'Married',
           2 : 'Single',
           3 : 'Single with a partner',
           4 : 'Separated/Widowed/Divorced'}


final_clusters_df['q49'].replace(Marital, inplace = True)



# Renaming Children
no_child = {1 : 'No child'}


final_clusters_df['q50r1'].replace(no_child, inplace = True)




# Renaming Children
yes_zero_child = {1 : 'Yes, under 6'}


final_clusters_df['q50r2'].replace(yes_zero_child, inplace = True)


# Renaming Children
yes_child = {1 : 'Yes, 6-12'}


final_clusters_df['q50r3'].replace(yes_child, inplace = True)


# Renaming Children
yes_one_child = {1 : 'Yes, 13-17'}


final_clusters_df['q50r4'].replace(yes_one_child, inplace = True)


# Renaming Children
yes_two_child = {1 : 'Yes, over 18'}


final_clusters_df['q50r5'].replace(yes_two_child, inplace = True)



# Renaming Race
race = {1 : 'White',
        2 : 'Black',
        3 : 'Asian',
        4 : 'Hawaiian',
        5 : 'Indian',
        6 : 'other'}

final_clusters_df['q54'].replace(race, inplace = True)




# Renaming Race
Hispanic = {1 : 'Yes',
            2 : 'No'}

final_clusters_df['q55'].replace(Hispanic, inplace = True)


# Renaming Income
income = {1 : 'Under $10,000',
          2 : '$10,000 - $14,999',
          3 : '$15,000 - $19,999',
          4 : '$20,000 - $29,999',
          5 : '$30,000 - $39,999',
          6 : '$40,000 - $49,999',
          7 : '$50,000 - $59,999',
          8 : '$60,000 - $69,999',
          9 : '$70,000 - $79,999',
          10 : '$80,000 - $89,999',
          11 : '$90,000 - $99,999',
          12 : '$100,000 - $124,999',
          13 : '$125,000 - $149,999',
          14 : '$150,000 and Over'}


final_clusters_df['q56'].replace(income, inplace = True)


# Renaming Gender
gender = {1 : 'Male',
          2 : 'Female'}

final_clusters_df['q57'].replace(gender, inplace = True)

###############################################################################
# Combining PCA and Clustering
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

customers_k_pca = KMeans(n_clusters = 8,
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


centroids_pca_df.columns = ['A little above mean', 'Middle Age Antisocial', 
                            'Tipycal Millenial','Apple lovers','Android lovers']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('customers_pca_centriods.xlsx')
                            


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


final_clusters_df = pd.concat([final_df.loc[ : , ['q1',
                                                  'q48',
                                                  'q49',
                                                  
                                                  'q50r1','q50r2','q50r3',
                                                  'q50r4','q50r5',
                                                  
                                                  'q54',
                                                  'q55',
                                                  'q56',
                                                  'q57'] ],
                               
                               clst_pca_df],
                               axis = 1)


print(final_clusters_df)



'''
Labeling the variables accordlying 
'''

# Renaming age
age = {1 : 'Under 18',
       2 : '18-24',
       3 : '25-29',
       4 : '30-34',
       5 : '35-39',
       6 : '40-44',
       7 : '45-49',
       8 : '50-54',
       9 : '55-59',
       10 : '60-64',
       11 : '65+'}


final_clusters_df['q1'].replace(age, inplace = True)


# Renaming Education
education = {1 : 'Some High School',
             2 : 'High School graduate',
             3 : 'Some College',
             4 : 'College Graduate',
             5 : 'Some Post-Graduate',
             6 : 'Post-Graduate Degree'}


final_clusters_df['q48'].replace(education, inplace = True)


# Renaming Marital Status
Marital = {1 : 'Married',
           2 : 'Single',
           3 : 'Single with a partner',
           4 : 'Separated/Widowed/Divorced'}


final_clusters_df['q49'].replace(Marital, inplace = True)



# Renaming Children
no_child = {1 : 'No child'}


final_clusters_df['q50r1'].replace(no_child, inplace = True)




# Renaming Children
yes_zero_child = {1 : 'Yes, under 6'}


final_clusters_df['q50r2'].replace(yes_zero_child, inplace = True)


# Renaming Children
yes_child = {1 : 'Yes, 6-12'}


final_clusters_df['q50r3'].replace(yes_child, inplace = True)


# Renaming Children
yes_one_child = {1 : 'Yes, 13-17'}


final_clusters_df['q50r4'].replace(yes_one_child, inplace = True)


# Renaming Children
yes_two_child = {1 : 'Yes, over 18'}


final_clusters_df['q50r5'].replace(yes_two_child, inplace = True)



# Renaming Race
race = {1 : 'White',
        2 : 'Black',
        3 : 'Asian',
        4 : 'Hawaiian',
        5 : 'Indian',
        6 : 'other'}

final_clusters_df['q54'].replace(race, inplace = True)




# Renaming Race
Hispanic = {1 : 'Yes',
            2 : 'No'}

final_clusters_df['q55'].replace(Hispanic, inplace = True)


# Renaming Income
income = {1 : 'Under $10,000',
          2 : '$10,000 - $14,999',
          3 : '$15,000 - $19,999',
          4 : '$20,000 - $29,999',
          5 : '$30,000 - $39,999',
          6 : '$40,000 - $49,999',
          7 : '$50,000 - $59,999',
          8 : '$60,000 - $69,999',
          9 : '$70,000 - $79,999',
          10 : '$80,000 - $89,999',
          11 : '$90,000 - $99,999',
          12 : '$100,000 - $124,999',
          13 : '$125,000 - $149,999',
          14 : '$150,000 and Over'}


final_clusters_df['q56'].replace(income, inplace = True)


# Renaming Gender
gender = {1 : 'Male',
          2 : 'Female'}

final_clusters_df['q57'].replace(gender, inplace = True)


# Adding a productivity step
data_df = final_clusters_df

########################
# Income
########################

# Apple lover
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y = 'Apple lovers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()



# Typical Millenial
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y = 'Tipycal Millenial',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


# Gender
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y = 'Apple lovers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()


# Typical Millenial
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y = 'Tipycal Millenial',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()




# Age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y = 'Android lovers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()