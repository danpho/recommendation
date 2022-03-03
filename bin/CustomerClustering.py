"""
This program is designed to cluster customers in different group,
so, the following analysis and recommendation can be developed.
This program is developed based on EDA analysis "h_m_eda_first_look_LW.py"
v0.0
currently, only choose features "age" and "postal_code" to cluster existing customers
User can define remove_lists to keep target features
this version cannot automatically identify and transform data type
ref:
whole picture of K-Means
https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/#:~:text=WCSS%20is%20the%20sum%20of,is%20largest%20when%20K%20%3D%201.
how to choose elbow point? Silhouette method
https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from matplotlib.cm import rainbow

# set paths
import platform
if platform.system() == "Windows":
    dir_separator = "\\"
else:
    dir_separator = "/"

work_dir = os.getcwd() + dir_separator
data_dir = work_dir + ".." + dir_separator \
           + "data" + dir_separator + "hm" + dir_separator
# data_dir = work_dir + dir_separator + "TechlentCamp" + dir_separator \
#     + "data" + dir_separator + "hm" + dir_separator
# test directory existing
if os.path.exists(data_dir):
    print("Existing data directory {}.".format(data_dir))
else:
    print("Not existing data directory {}".format(data_dir))

# user settings
# set K range
k_start = 2
k_end = 40
# user can define which features can be kept
# remove_lists = []
# remove_lists = ['customer_id', 'FN', 'Active', 'club_member_status','fashion_news_frequency']
remove_lists = ['customer_id', 'FN', 'Active', 'club_member_status','fashion_news_frequency', 'postal_code']

# read data
csv_file = "customers.csv"
file_name = os.path.join(data_dir,csv_file)
print("reading csv file {}...".format(file_name))
customers = pd.read_csv(file_name)
# print(customers.head())

# clean data based on EDA
print("cleaning imported data ...")
customers['FN'] = customers['FN'].fillna(0)
customers['Active'] = customers['Active'].fillna(0)
customers['club_member_status'] = customers['club_member_status'].fillna('NON_MEMBER')
customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NONE')
customers['fashion_news_frequency'] = customers['fashion_news_frequency'].replace('None', 'NONE')

customers.dropna(inplace=True)
len(customers['age'])

# all

# KMeans
# prepared training data for clustering
print("Applying KMeans...")
features = list(customers.columns)
for element in remove_lists:
    features.remove(element)
print("kept features are :\n", features)

# transform target features to numeric type
# postal_code is object, here only use a series of number to replace the object
zip_set = set(customers.postal_code.unique())
i = 0
zip_dict = {}
for zip in zip_set:
    i += 1
    zip_dict[zip] = i

customers["postal_code"] = customers["postal_code"].apply(lambda x: zip_dict[x])

# apply KMeans
X = customers[features]
# customers = customers.reset_index()
# max_age = max(customers.age)
# X = customers[features].apply(lambda x: x/max_age)
# wcss is the sum of squared distance between each point and the centroid in a cluster
wcss = []   # within-cluster sum of square
sil = []    # use SilHouette method to find the elbow point K

wcss_csv = os.path.join(data_dir, "hm_customers_wcss.csv")
for i in range(k_start, k_end+1):
    print("Estimating the k={}/{} fitting...".format(i, k_end-k_start+1))
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=2022)
    kmeans.fit(X)
    labels = kmeans.labels_
    print("\tappending kmeans.inertia_ to wcss...")
    wcss.append(kmeans.inertia_)
    # print("\tappending silhouette_score to sil...")
    # sil.append(silhouette_score(X, labels, metric='euclidean'))
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

print("Export wcss into csv file {}".format(wcss_csv))
ddf = pd.DataFrame(wcss)
ddf.to_csv(wcss_csv, header=["css"], sep=",")

# section: visualization
plt.figure(figsize=(12, 12))

# plot elbow curve
plt.plot(np.arange(k_start, k_end+1), wcss)
plt.scatter(np.arange(k_start, k_end+1), wcss)
plt.xlabel("Clusters")
plt.ylabel('SSE')
_ = file_name.split("\\")[-1]
plt.title('Elbow curve: SSE(K) of {}'.format(_))
new_file_name = data_dir + "SSE_" + csv_file
print(new_file_name)
pdf = new_file_name + ".pdf"
png = new_file_name + ".png"
print("Saving figures in {}:\n{}\n{}".format(data_dir, pdf, png))
# jpg = new_file_name + ".jpg"
# plt.savefig(pdf, dpi=600, format="pdf")
# plt.savefig(png, dpi=600, format="png")

plt.show()

# Apply Silhouette method to find the best K
# The silhouette value measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation).
from sklearn.metrics import silhouette_score
# k_best = max(sil)
# print("The best in elbow figure is K={}".format(k_best))
#
# kmeans = KMeans(n_clusters=k_best, init="k-means++", rangdom_state=2022)
# y_kmeans = kmeans.fit_predict(X)
# colors = rainbow(range(k_best))
# for i in range(k_best):
#     lbl = "Cluster" + str(i)
#     plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=60, c=colors[i])
#     plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
#     plt.xlabel('Annual Income (k$)')
#     plt.ylabel('Spending Score (1-100)')
#     plt.legend()


# plt.show()