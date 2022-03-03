"""
This program is designed to save centroids, labeled data
input file: files (csv_file) are used to for clustering
output_file: centroid file, labelled file
centroids_keywords + csv_file

version:
v0.0
grouped by "age" may by "age + postal_code"
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans

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

csv_file = input("Input name of csv file without labels: ")
# centroids_file = data_dir + "centroids_" + csv_file
# labeled_csv_file = data_dir + 'labeled_' + csv_file
centroids_file = data_dir + "centroids_age_" + csv_file
labeled_csv_file = data_dir + 'labeled_age_' + csv_file
csv_file = data_dir + csv_file

print("Input K range below.")
k_min = input("Input floor value of K: ")
k_max = input("Input ceiling value of K: ")
k_min, k_max = int(k_min), int(k_max)
if k_min > k_max: k_min, k_max = k_max, k_min
print("K range is [{},{}].".format(k_min, k_max))

flag = True
if os.path.exists(csv_file):
    print("{} exist. Will cluster data.".format(csv_file))
else:
    print("{} file does NOT exist.")
    flag = False

print("Reading data ...")
df = pd.read_csv(csv_file)
# features = ["age", "postal_code"]
features = ["age"]
X = df[features]
# X = df[["age"]]
# X = df[["age", "postal_code"]]

if flag:
    for k in range(k_min, k_max+1):
        print("Fitting k={} ...".format(k))
        # fit X
        kmeans_optimum = KMeans(n_clusters=k, init='k-means++', random_state=2022)
        # predict y
        y = kmeans_optimum.fit_predict((X))
        print(np.unique(y))
        # cluster = "cls_" + str(k)
        cluster = "age_cls_" + str(k)
        df[cluster] = y
        # print(df.head(2))

        # dataframe for cluster_centers
        df_sub = pd.DataFrame(kmeans_optimum.cluster_centers_)
        # sort by keywords
        df_sub = df_sub.sort_values(by=[2])
        df_centroids[c_x] = df_sub[0]
        df_centroids[c_y] = df_sub[1]
        df_centroids[c_z] = df_sub[2]
        df_sub = df_sub.iloc[0:0]   # empty the dataframe

df_centroids.to_csv(centroids_file, sep=',', columns=df_centroids.columns, index=False)

print(df)

df.to_csv(labeled_csv_file, sep=',', columns=df.columns, index=False)
print("Saved labeled data to csv file {}".format(labeled_csv_file))
print("Saved centroids to csv file {}".format(centroids_file))