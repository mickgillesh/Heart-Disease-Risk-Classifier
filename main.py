import numpy as np
import pandas as pd
import math


import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn import cluster
from sklearn.preprocessing import scale
from sklearn.preprocessing import OrdinalEncoder

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7,7)
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

data = pd.read_csv('C://Users//micha//OneDrive//Documents//UniWork//CS982- Big Data Technologies//Coursework//Assignment//heart.csv')

#Remove any rows with 0 values in the continuous measurements that cannot be 0
data[["Age","RestingBP","Cholesterol","MaxHR"]] = data[["Age","RestingBP","Cholesterol","MaxHR"]].replace(0,np.nan)
data = data.dropna(subset=["Age","RestingBP","Cholesterol","MaxHR"])
outcome = data.HeartDisease
features = data.drop(["HeartDisease"], axis = 1)

#Use bar graphs to show those with and without heart disease for catagorical features
cat_features = ["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope","FastingBS"]
cat_features_names = ["Sex","Chest Pain Type","Resting ECG","Exercise Angina","ST Slope", "Fasting Blood Sugar"]
i=0
for cat in cat_features:
    hd_patch = mpatches.Patch(label='No Heart Disease')
    nhd_patch = mpatches.Patch(color='orange', label='Heart Disease')

    data.groupby([cat, 'HeartDisease']).size().unstack().plot(kind='bar', stacked=True)
    plt.legend(handles=[hd_patch, nhd_patch])
    plt.title('Number of Indeviduals grouped by ' +cat_features_names[i]+ ' and Heart Disease')
    i=i+1
    plt.show()

#Plot the total number of heart disease and no heart disease instances on a bar chart
count_CVD = data['HeartDisease'].value_counts()
count_sex = data['Sex'].value_counts()
labels_CVD = ('Heart Disease', 'No Heart Disease')
plt.title('Total With and Without Heart Disease')
plt.bar(labels_CVD, count_CVD, align='center', color=['DarkOrange','blue'],width=0.5)
plt.show()


#use ordinal encoder to assign int values to the string labels for categotical features
enc = OrdinalEncoder()
enc.fit(features[["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"]])
features[["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]] = enc.transform(features[["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"]])


#Show density functions with boxplot subplots of each continuous measurement for those with and without heart disease
df = pd.concat([features, outcome], axis=1)
num_features = [df.groupby(["HeartDisease"]).RestingBP,df.groupby(["HeartDisease"]).Cholesterol,df.groupby(["HeartDisease"]).MaxHR,df.groupby(["HeartDisease"]).Oldpeak,df.groupby(["HeartDisease"]).Age]
num_feature_names = ["RestingBP","Cholesterol","MaxHR","Oldpeak", "Age"]

grouped = data.groupby(["HeartDisease"])
group_CVD = grouped[["RestingBP","Cholesterol","MaxHR","Oldpeak", "Age"]].get_group(0)
group_no_CVD = grouped[["RestingBP","Cholesterol","MaxHR","Oldpeak", "Age"]].get_group(1)
i=0
for feat in num_features:
    #Create the density plot
    fig = plt.figure(1)
    fig.add_subplot(3, 1, (1, 2))
    plot = feat.plot.kde()
    hd_patch = mpatches.Patch(label='No Heart Disease')
    nhd_patch = mpatches.Patch(color='orange', label='Heart Disease')
    plt.legend(handles=[hd_patch, nhd_patch])
    plot_mode = feat.median()
    plt.title('Density and box plot showing distributions of ' + num_feature_names[i])
    x_max = max(feat.max())
    x_min = min(feat.min()) - (0.1*x_max)
    x_max = max(feat.max()) + (0.1*x_max)
    plt.xlim(x_min,x_max)
    #Add a subplot below the density plot and then create box plots for those with and without heart disease
    fig.add_subplot(3, 1, 3)
    no_CVD = plt.boxplot(group_no_CVD[num_feature_names[i]],vert=False, positions=[1],
                     boxprops=dict(color="darkorange"),
                     capprops=dict(color="darkorange"),
                     whiskerprops=dict(color="darkorange"),
                     flierprops=dict(color="darkorange", markeredgecolor="darkorange"),
                     medianprops=dict(color="darkorange"))
    plt.xlim(x_min, x_max)
    plt.xlabel(num_feature_names[i])
    plt.yticks([1, 2], ['Heart \n Disease', 'No Heart \n Disease'])
    CVD = plt.boxplot(group_CVD[num_feature_names[i]],vert=False, positions=[2],
                  boxprops=dict(color="blue"),
                  capprops=dict(color="blue"),
                  whiskerprops=dict(color="blue"),
                  flierprops=dict(color="blue", markeredgecolor="blue"),
                  medianprops=dict(color="blue"))
    plt.xlim(x_min, x_max)
    plt.xlabel(num_feature_names[i])
    plt.yticks([1, 2], ['Heart \n Disease', 'No Heart \n Disease'])
    i=i+1
    plt.show()


#Split the data into train and test datasets
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(features, outcome, test_size=0.2,random_state= 1)
Y_train = np.array(Y_train)

#Balance the train datset
oversample = SMOTE(random_state=1)
X_train, Y_train = oversample.fit_resample(X_train, Y_train)


#Scale the numeric values
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

#Apply logistic regression with RFECV to create model with optimal number of variables
lm = LogisticRegression(max_iter = math.inf)
rfecv = RFECV(lm)
rfecv.fit(X_train, Y_train)
print(rfecv.ranking_)

#Scale the test data and then predict
scaler = StandardScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)
predicted = rfecv.predict(X_test)

#Print classification performance metrics
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))

#Create visualisation of predicted outcome performance
plt.clf
plt.figure(10)
#Display the true classification of each datapoint
X_test_reduced = TSNE(n_components= 2, perplexity= 10).fit_transform(X_test)
plt.scatter(X_test_reduced[np.where(Y_test == 0)[0],0],X_test_reduced[np.where(Y_test == 0)[0],1],marker = 'x')
plt.scatter(X_test_reduced[np.where(Y_test == 1)[0],0],X_test_reduced[np.where(Y_test == 1)[0],1],marker = 'o')

plt.figure(11)
#Plot all predicted data points categorising each point by whether it was correct or not
plt.scatter(X_test_reduced[np.where(predicted == 0)[0],0],X_test_reduced[np.where(predicted == 0)[0],1],marker = 'x')
plt.scatter(X_test_reduced[np.where(predicted == 1)[0],0],X_test_reduced[np.where(predicted == 1)[0],1],marker = 'o')
plt.scatter(X_test_reduced[np.where((predicted == 1) & (Y_test ==0))[0],0],X_test_reduced[np.where((predicted == 1) & (Y_test ==0))[0],1],marker = 'o',color = 'green')
plt.scatter(X_test_reduced[np.where((predicted == 0) & (Y_test ==1))[0],0],X_test_reduced[np.where((predicted == 0) & (Y_test ==1))[0],1],marker = 'x',color = 'red')
tp = mlines.Line2D([],[],label='True Positive', color = 'orange',marker='o', linewidth= 0  )
tn = mlines.Line2D([],[],label = 'True Negative',color = 'blue',marker = 'x',linewidth= 0)
fp = mlines.Line2D([],[],label='False Positive', color = 'green',marker='o' ,linewidth= 0)
fn = mlines.Line2D([],[],label = 'False Negative',color = 'red',marker = 'x',linewidth= 0)
plt.legend(handles=[tp,tn,fp,fn])
plt.title('Scatter Plot of Predicted Results')
plt.show()

#Clustering into 2 clusters using scaled values using hierarchical(not used in final project)
features = scale(features)
model = cluster.AgglomerativeClustering(n_clusters=2, linkage="average", affinity ="cosine")
model.fit(features)

print("H Completeness score", metrics.completeness_score(data.HeartDisease, model.labels_))
print("H Homogeneity score", metrics.homogeneity_score(data.HeartDisease, model.labels_))
# print("silhouette score", metrics.silhouette_score(outcome, model.labels_))

#clustering using K means
kmeans = cluster.KMeans(n_clusters=2,n_init=1000)
kmeans.fit(features)
print("Kmeans Completeness score", metrics.completeness_score(data.HeartDisease, kmeans.labels_))
print("Kmeans Homogeneity score", metrics.homogeneity_score(data.HeartDisease, kmeans.labels_))
print("Kmeans Silhouette score", metrics.silhouette_score(np.array(outcome).reshape(-1,1), model.labels_))
print("Kmeans Accuracy score", metrics.accuracy_score(data.HeartDisease, kmeans.labels_))

#Create a reduced dimensions plot of clusters along side the true classifications
features_reduced = TSNE(n_components= 2, perplexity= 10).fit_transform(features)
fig = plt.figure(12)
fig.add_subplot(1 ,2, 1)
plt.title('K Means Clustering')
plt.scatter(features_reduced[np.where(kmeans.labels_ == 0)[0],0],features_reduced[np.where(kmeans.labels_ == 0)[0],1],marker = 'x',c='green')
plt.scatter(features_reduced[np.where(kmeans.labels_ == 1)[0],0],features_reduced[np.where(kmeans.labels_ == 1)[0],1],marker = 'o',c='red')
fig.add_subplot(1 ,2, 2)
plt.title('True Classification')
plt.scatter(features_reduced[np.where(outcome == 0)[0],0],features_reduced[np.where(outcome == 0)[0],1],marker = 'x')
plt.scatter(features_reduced[np.where(outcome == 1)[0],0],features_reduced[np.where(outcome == 1)[0],1],marker = 'o')


#Dendogram plot (Implemented but not used in project)
plt.show()
from scipy.cluster.hierarchy import dendrogram, linkage
model = linkage(features, 'ward')
plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(model, leaf_font_size=8.,)
plt.show()


