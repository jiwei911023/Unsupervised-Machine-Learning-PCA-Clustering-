# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:08:40 2019

@author: Jiwei li
"""
###########################################################
##########################Final############################
###########################################################
print("""
      This code is to analysis the apps download behavior and other important
      informations related to the consumers.
      """)

# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis
from sklearn.cluster import KMeans # k-means clustering



# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)



# Importing dataset
mobile = pd.read_excel('finalExam_Mobile_App_Survey_Data_final_exam-2.xlsx')
###############################################################################
# Code For Data Analysis(EDA)
###############################################################################

mobile.info()

mobile.head(n = 5)

mobile.describe().round(2)

mobile.columns

# Printing our columns with indexes
for col in enumerate(mobile):
    print(col)


df_percentiles = mobile.describe(percentiles = [0.01,
                                                      0.05,
                                                      0.10,
                                                      0.25,
                                                      0.50,
                                                      0.75,
                                                      0.90,
                                                      0.95,
                                                      0.99]).round(2)



df_percentiles.loc[['min',
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

#Check the missing value
mobile.isnull().sum()


# Viewing the first few rows of the data
mobile.head(n = 5)





"""
Basic information:
    * 88 variables of type int64
    * no missing values
    * 1552 observations
"""

########################
# Correlation analysis
########################

fig, ax = plt.subplots(figsize = (8, 8))


df_corr = mobile.corr().round(2)


sns.heatmap(df_corr,
            cmap = 'coolwarm',
            square = True,
            annot = True)


plt.savefig('mobile_app_correlations.png')
plt.show()

#####Histgram for some important variables

#Gender&Income
fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(2, 1, 1)
sns.distplot(a = mobile['q57'],
             hist = True,
             kde = True,
             rug=True,
             color = 'blue')
plt.xlabel("Gender")



plt.subplot(2, 1, 2)
sns.distplot(a = mobile['q56'],
             hist = True,
             kde = True,
             rug=True,
             color = 'red')
plt.xlabel("Income")



plt.show()

#Marital&Children
fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(2, 1, 1)
sns.distplot(a = mobile['q49'],
             hist = True,
             kde = True,
             rug=True,
             color = 'blue')
plt.xlabel("Marital")



plt.subplot(2, 1, 2)
sns.distplot(a = mobile['q56'],
             hist = True,
             kde = True,
             rug=True,
             color = 'red')
plt.xlabel("Have_Children")



plt.show()


#######################################################
###################Model Code##########################
#######################################################

#################### 1 .PCA###########################


########################
# Step 1: Remove demographic information
########################
# Printing our columns with indexes
for col in enumerate(mobile):
    print(col)
    
mobile_features_reduced = mobile.iloc[ : , 2:77]

print("""
      Before we start the pca, we should do a little bit modification on 
      dataset , so that the result will be more logical and accepetable.
      """)

often = {1:4,
         2:3,
         3:2,
         4:1}

mobile.iloc[:,25:37].replace(often,inplace=True)

########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(mobile)


X_scaled = scaler.transform(mobile)

X_scaled_df=pd.DataFrame(X_scaled)



########################
# Step 3: Run PCA without limiting the number of components
########################

mobile_pca = PCA(n_components = None,
                 random_state = 508)


mobile_pca.fit(X_scaled)


pca_factor_strengths = mobile_pca.transform(X_scaled)




########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(mobile_pca.n_components_)


plt.plot(features,
         mobile_pca.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'red')


plt.title('Reduced Moblie Apps Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()

print(f"""
      Right now we have the following:
      Principal Components: {(mobile_pca.explained_variance_ratio_[0] + 
                              mobile_pca.explained_variance_ratio_[1] + 
                              mobile_pca.explained_variance_ratio_[2] +
                              mobile_pca.explained_variance_ratio_[3]
                              
                                                            ).round(2)}

""")


########################
# Step 5: Run PCA again based on the desired number of components
########################

mobile_pca_reduced = PCA(n_components = 4,
                           random_state = 508)


mobile_pca_reduced.fit(X_scaled)



########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(mobile_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(mobile.columns)


print(factor_loadings_df)


factor_loadings_df.to_excel('mobile_factor_loadings1.xlsx')

########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = mobile_pca_reduced.transform(X_scaled)


X_pca_df = pd.DataFrame(X_pca_reduced)
########################
# Step 8: Rename your principal components and reattach demographic information
########################
X_pca_df.columns = ['User_Preference', 
                    'Customer_Usage', 
                    'Apps',
                    'User_Information']


##########Feature Engineering###############

##We can make some combination on the columns which is related to age
No_child = {1 :0}  

mobile['q50r1'].replace(No_child, inplace = True)

mobile['child_info'] = mobile.apply(lambda x: x['q50r1']+x['q50r2'] + x['q50r3'] + x['q50r4']
+ x['q50r5'] , axis = 1)


###############Create the final dataset
final_pca_df = pd.concat([mobile.loc[ : , ['q1', 'q48',  'q49', 
                                           'child_info', 'q54', 
                                           'q55', 'q56','q57']] , 
                                            X_pca_df], axis = 1)

#######################################################################
# Step 10: Analyze in more detail(transform the demographic categorical 
#variables into their real meanings!!)
#######################################################################


# Renaming Age
Age = {1 : 'Under-18',
       2 : '18-24',
       3 : '25-29',
       4 : '30-34',
       5 : '35-39',
       6 : '40-44',
       7 : '45-49',
       8 : '50-54',
       9 : '55-59',
       10 : '60-64',
       11 : 'over_65'}

final_pca_df['q1'].replace(Age, inplace = True)



# Renaming education
education = {1 : 'some_high_school',
             2 : 'high_school',
             3 : 'some_college',
             4 : 'college',
             5 : 'some_PG',
             6 : 'PG'}


final_pca_df['q48'].replace(education, inplace = True)



# Renaming marital
marital = {1 : 'married',
           2 : 'single',
           3 : 'partner',
           4 : 'divorced'}


final_pca_df['q49'].replace(marital, inplace = True)



# Renaming children
child = {0 : 'no_child',
           1 : 'have_child',
           2 : 'have_child',
           3 : 'have_child',
           4 : 'have_child'}


final_pca_df['child_info'].replace(child, inplace = True)


# Renaming race
race = {1 : 'white',
        2 : 'black',
        3 : 'asian',
        4 : 'hawaiian',
        5 : 'A_indian',
        6 : 'other'}


final_pca_df['q54'].replace(race, inplace = True)



# Renaming Hispanic/Latino
HisLat = {1 : 'Yes',
          2 : 'No'}

final_pca_df['q55'].replace(HisLat, inplace = True)



# Renaming Income Level
Income = {1 : 'under_10K',
          2 : '10K-15K',
          3 : '15K-20K',
          4 : '20K-30K',
          5 : '30K-40K',
          6 : '40K-50K',
          7 : '50K-60K',
          8 : '60K-70K',
          9 : '70K-80K',
          10 : '80K-90K',
          11 : '90K-100K',
          12 : '100K-125K',
          13 : '125K-150K',
          14 : 'over_150K'}

final_pca_df['q56'].replace(Income, inplace = True)



# Renaming Gender
Gender = {1 : 'Male',
          2 : 'Female'}


final_pca_df['q57'].replace(Gender, inplace = True)

#################################################
############Use boxplot to analyze ##############
#################################################

# Analyzing by age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'User_Preference',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Customer_Usage',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Apps',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'User_Information',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()






# Analyzing by education
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'User_Preference',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Customer_Usage',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Apps',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'User_Information',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()

# Analyzing by marital
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'User_Preference',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'Customer_Usage',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'Apps',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'User_Information',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



# Analyzing by child
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'child_info',
            y =  'User_Preference',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'child_info',
            y =  'Customer_Usage',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'child_info',
            y =  'Apps',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'child_info',
            y =  'User_Information',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



# Analyzing by race
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y =  'User_Preference',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y =  'Customer_Usage',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y =  'Apps',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y =  'User_Information',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



# Analyzing by Hispanic/Latino
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y =  'User_Preference',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y =  'Customer_Usage',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y =  'Apps',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y =  'User_Information',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()

# Analyzing by Income
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'User_Preference',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Customer_Usage',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Apps',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'User_Information',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()


# Analyzing by gender
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'User_Preference',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Customer_Usage',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Apps',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'User_Information',
            data = final_pca_df)

plt.ylim(-2, 3)
plt.tight_layout()
plt.show()



################################################################
##################Combining PCA and Clustering##################
################################################################


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



######################################################
# Step 3: Experiment with optimal numbers of clusters
#####################################################

###############################################################################
# Plotting Intertia
###############################################################################

"""
    How many clusters do we need? Which number of clusters is 'best' for our
    data? These questions can be answered using the metric inertia.
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

#################n_cluster=5####################
mobile_k_pca = KMeans(n_clusters = 5,
                   random_state = 508)


mobile_k_pca.fit(X_pca_clust_df)


mobile_kmeans_pca = pd.DataFrame({'cluster': mobile_k_pca.labels_})


print(mobile_kmeans_pca.iloc[: , 0].value_counts())




########################
# Step 4: Analyze cluster centers
########################
X_pca_df.columns
centroids_pca = mobile_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['User_Preference', 'Customer_Usage', 'Apps','User_Information']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('mobile_pca_centriods1.xlsx')



########################
# Step 5: Analyze cluster memberships
########################

clst_pca_df = pd.concat([mobile_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)



########################
# Step 6: Reattach demographic information
########################

final_pca_clust_df = pd.concat([mobile.loc[ : , ['q1', 'q48',  'q49', 'child_info', 
                                                'q54', 'q55', 'q56','q57']],
                                                 clst_pca_df], axis = 1)


print(final_pca_clust_df.head(n = 5))



#######################################################################
# Step 7: Analyze in more detail(transform the demographic categorical 
#variables into their real meanings!!)
#######################################################################

# Renaming Age
Age = {1 : 'Under-18',
       2 : '18-24',
       3 : '25-29',
       4 : '30-34',
       5 : '35-39',
       6 : '40-44',
       7 : '45-49',
       8 : '50-54',
       9 : '55-59',
       10 : '60-64',
       11 : 'over_65'}

final_pca_df['q1'].replace(Age, inplace = True)



# Renaming education
education = {1 : 'some_high_school',
             2 : 'high_school',
             3 : 'some_college',
             4 : 'college',
             5 : 'some_PG',
             6 : 'PG'}


final_pca_df['q48'].replace(education, inplace = True)



# Renaming marital
marital = {1 : 'married',
           2 : 'single',
           3 : 'partner',
           4 : 'divorced'}


final_pca_df['q49'].replace(marital, inplace = True)



# Renaming child
child = {0 : 'no_child',
           1 : 'have_child',
           2 : 'have_child',
           3 : 'have_child',
           4 : 'have_child'}


final_pca_df['child_info'].replace(child, inplace = True)


# Renaming race
race = {1 : 'white',
        2 : 'black',
        3 : 'asian',
        4 : 'hawaiian',
        5 : 'A_indian',
        6 : 'other'}


final_pca_df['q54'].replace(race, inplace = True)



# Renaming Hispanic/Latino
HisLat = {1 : 'Yes',
          2 : 'No'}

final_pca_df['q55'].replace(HisLat, inplace = True)



# Renaming Income Level
Income = {1 : 'under_10K',
          2 : '10K-15K',
          3 : '15K-20K',
          4 : '20K-30K',
          5 : '30K-40K',
          6 : '40K-50K',
          7 : '50K-60K',
          8 : '60K-70K',
          9 : '70K-80K',
          10 : '80K-90K',
          11 : '90K-100K',
          12 : '100K-125K',
          13 : '125K-150K',
          14 : 'over_150K'}

final_pca_df['q56'].replace(Income, inplace = True)



# Renaming Sex
Gender = {1 : 'Male',
          2 : 'Female'}


final_pca_df['q57'].replace(Gender, inplace = True)


###create final dataset which contain the clusters and components#########

mobile2 = final_pca_clust_df.copy()

#######Export the dataframe to excel file

mobile2.to_excel("ffinal_mobile2.xlsx")


######################Final boxplot#######################


########################
# sex
########################
X_pca_df.columns

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y = 'User_Preference',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y = 'Customer_Usage',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y = 'Apps',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y = 'User_Information',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()



########################
# income
########################


fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y = 'User_Preference',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y = 'Customer_Usage',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y = 'Apps',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y = 'User_Information',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()



########################
# education
########################

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y = 'User_Preference',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y = 'Customer_Usage',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y = 'Apps',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y = 'User_Information',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()


########################
# age
########################

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y = 'User_Preference',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y = 'Customer_Usage',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y = 'Apps',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y = 'User_Information',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()


########################
# race
########################

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y = 'User_Preference',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y = 'Customer_Usage',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y = 'Apps',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y = 'User_Information',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()



########################
# marital
########################

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y = 'User_Preference',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y = 'Customer_Usage',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y = 'Apps',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y = 'User_Information',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

########################
# children
########################

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'child_info',
            y = 'User_Preference',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'child_info',
            y = 'Customer_Usage',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'child_info',
            y = 'Apps',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'child_info',
            y = 'User_Information',
            hue = 'cluster',
            data = mobile2)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

X_pca_df.comlumns
#######################plot the clusters value###################
#Attibute mean :user information/preferrences/apps/usage
clu = mobile_k_pca.cluster_centers_
x = [1,2,3,4]  
colors = ['red','green','yellow','blue']  
for i in range(4):  
   plt.plot(x,clu[i],label='cluster'+str(i),linewidth=6-i,color=colors[i],marker='o')   
plt.xlabel('Attribute')  
plt.ylabel('values')  
plt.show()

