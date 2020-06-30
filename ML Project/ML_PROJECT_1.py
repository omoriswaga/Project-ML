import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV,ShuffleSplit
import seaborn as sm
from sklearn import tree
from sklearn.svm import SVR

# Defining at first the data_frame
df = pd.read_csv("bengaluru_house_prices.csv")

# shape of the data frame
# print(df.shape) which is (13320, 9)
# print(df.groupby('area_type')['area_type'].agg('count'))

# I will make an assumption that availability does don matter in determining
# the price of an house
# so i will drop availability column
df.drop("availability",1,inplace=True)
# Data cleaning process
# First process is handling na values

# Society has 5502 nan value
# so i will drop society column from my data frame (This is my personal choice)
df.drop('society',1,inplace=True)
# I will fill balcony nan values (609) with zero (my personal choice)
df.balcony = df.balcony.fillna(0)
# I will drop the other nan rows
df.dropna(inplace=True)
# print(df.isnull().sum()) to check for null values
# print(df.groupby('size').describe())

# Applying a little lambda function to split our size column to an integer value only
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
# Drop size column
df.drop(['size'],1,inplace=True)
# A simple fuction to check the total_sqft column for data errors
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
df1 = df[~df.total_sqft.apply(lambda x: is_float(x))]
# A simple fuction that deals with total_sqft data erroes (my own choice of handling the error)
def split_average(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return (float(x))
    except:
        return None
df["total_sqft"] = df.total_sqft.apply(split_average)
# print(df.isnull().sum())
# total_sqft column has a total of 46 nan values, so i will drop them all
df.dropna(inplace=True)

# Some little feature engineering
# Made a price per sqft column (dividing price by the corresponding sqft)
df2 = df.copy()
df2["price_per_sqft"] = df2.price*100000/df2.total_sqft
# To show the amount of unique location and how many rows the have
# print(len(df2.location.unique()))
location_stat = df2.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stat_less_than_10 = location_stat[location_stat<=10]
def location_cleaning(x):
    if x in location_stat_less_than_10:
        return "others"
    return x
df2.location = df2.location.apply(lambda x: location_cleaning(x))
# Dealing with outliers for the price_per_sqft column using the IQR method
Q1 = df2.price_per_sqft.quantile(0.25)
Q3 = df2.price_per_sqft.quantile(0.75)
IQR = Q3-Q1
min_threshold = Q1 - (1.5*IQR)
max_threshold = Q3 + (1.5*IQR)
df2 = df2[(df2.price_per_sqft>min_threshold) & (df2.price_per_sqft<max_threshold)]
# Also the typical sqft per bedroom is usually 300sqft_per_bedroom
# I will remove any outliers that not not fall under that threshold
df2 = df2[ (df2.total_sqft/df2.bhk)>300 ]
# creating a simple fuction to see error data points
def visualization(df,location):
    bhk2 = df[(df.location == location) & (df.bhk==2)]
    bhk3 = df[(df.location == location) & (df.bhk==3)]
    plt.scatter(bhk2.total_sqft,bhk2.price, marker="+",color='green')
    plt.scatter(bhk3.total_sqft,bhk3.price, color='red')
    plt.xlabel("Total square feet")
    plt.ylabel("Price")
    plt.show()
# visualization(df2,'Rajaji Nagar')
def bhk_outlier_removal(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby("location"):
        bhk_stat = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stat[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stat.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
# creating a new data frame
df3 = bhk_outlier_removal(df2)
# Removing the bathroom outliers
df3 = df3[df3.bath<df3.bhk + 2]
# drop unnecessary features
df3.drop(['price_per_sqft','balcony'],1,inplace=True)
# applying one hot encoding
area_type_dummies = pd.get_dummies(df3.area_type)
df3 = pd.concat([df3,area_type_dummies],1)
location_dummies = pd.get_dummies(df3.location)
df3 = pd.concat([df3,location_dummies],axis='columns')
df3.drop(['area_type','location','others','Built-up  Area'],axis='columns',inplace=True)
# Start getting features and target variables
X = df3.drop(['price'],axis='columns')
y = df3.price

# Using Hyper parameter tunning
#hp = {
#    'Linear Regression': {
#        'model': linear_model.LinearRegression(),
#        'params': {
#            'normalize': [True,False],
#        }
#},
#    'Decision tree': {
#        'model': tree.DecisionTreeRegressor(),
#        'params': {
#            'criterion': ["mse", "friedman_mse", "mae"],
#            'splitter': ["best", "random"]
#        }
# }
# }
#cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#scores=[]
#for model_name,info in hp.items():
#    model = GridSearchCV(info['model'],info['params'],cv=10,return_train_score=False)
#    model.fit(X,y)
#    scores.append({
#        'model name': model_name,
#        'Best score': model.best_score_,
#        'Best param': model.best_params_
#    })
#print(scores)
# [{'model name': 'Linear Regression', 'Best score': 0.7517020697496848, 'Best param': {'normalize': False}}]
# [{'model name': 'Decision tree', 'Best score': 0.7431048625618544, 'Best param': {'criterion': 'friedman_mse', 'splitter': 'random'}}]

# so in conclusion, Linear Regression performs better
model = linear_model.LinearRegression(normalize=False)
model.fit(X,y)
def predict_price(area,location,sqft,bath,bhk):
    loc_index_location = np.where(X.columns==location)[0][0]
    loc_index_area = np.where(X.columns==area)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if (loc_index_location >=6) & (loc_index_area>=3) :
        x[loc_index_location]=1
        x[loc_index_area] = 1
    return model.predict([x])
print(predict_price('Carpet  Area','1st Phase JP Nagar',1000,4,4))

#Saving the model on a pickle file
with open('bengaluru_house_prices_model.pickle','wb') as file:
    pickle.dump(model,file)

#Exporting my column feature structure to a json file
columns = {
    'coloumns_info': []
}
for x in X.columns:
    for key in columns:
        columns[key].append(x.lower())
with open('columns.json','w') as file:
    file.write(json.dumps(columns))
