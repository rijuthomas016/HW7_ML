from surprise import Dataset
from surprise import Reader
import os
import pandas as pd

file_path = os.path.expanduser('restaurant_ratings.txt')

reader = Reader(line_format='user item rating timestamp', sep='\t')

data = Dataset.load_from_file(file_path,reader=reader)

from surprise import SVD
from surprise.model_selection import cross_validate

algo = SVD()

cross_validate(algo, data, measures=['RMSE','MAE'], cv=3, verbose=True)

#PMF
algo = SVD(biased=False)

cross_validate(algo, data, measures=['RMSE','MAE'], cv=3, verbose=True)

from surprise import NMF

#NMF (non-negative matrix factorization) algorithm
algo = NMF()

from surprise import KNNBasic

#user based collaborative filtering
algo = KNNBasic(sim_options={'user_based':True})

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

#item based collaborative filtering
algo = KNNBasic(sim_options={'user_based':False})

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

import random
import numpy as np
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

algo = SVD()

cross_validate(algo, data, measures=['RMSE','MAE'], cv=3, verbose=True)

#PMF
algo = SVD(biased=False)

cross_validate(algo, data, measures=['RMSE','MAE'], cv=3, verbose=True)

#NMF (non-negative matrix factorization) algorithm
algo = NMF()

cross_validate(algo, data, measures=['RMSE','MAE'], cv=3, verbose=True)

#user based collaborative filtering
algo = KNNBasic(sim_options={'user_based':True})

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

#item based collaborative filtering
algo = KNNBasic(sim_options={'user_based':False})

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

import matplotlib.pyplot as plt

def problem14():
    plotRMSE= []
    plotMAE = []
    print("-----MSD similarity in User based Collaborative Filtering----")
    algo = KNNBasic(sim_options={'name':'MSD', 'user_based': True})
    user_MSD = cross_validate(algo, data, cv=3, verbose=False)
    plotRMSE.append(["User-based filtering", 1, user_MSD["test_rmse"].mean()])
    plotMAE.append(["User-based filtering", 1, user_MSD["test_mae"].mean()])

    print("-----Cosine similarity in User based Collaborative Filtering----")
    algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    user_COS = cross_validate(algo, data, cv=3, verbose=False)
    plotRMSE.append(["User-based filtering", 2, user_COS["test_rmse"].mean()])
    plotMAE.append(["User-based filtering", 2, user_COS["test_mae"].mean()])

    print("-----Pearson similarity in User based Collaborative Filtering----")
    algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': True})
    user_Pearson = cross_validate(algo, data, cv=3, verbose=False)
    plotRMSE.append(["User-based filtering", 3, user_Pearson["test_rmse"].mean()])
    plotMAE.append(["User-based filtering", 3, user_Pearson["test_mae"].mean()])

    print("-----MSD similarity in Item based Collaborative Filtering----")
    algo = KNNBasic(sim_options={'name':'MSD', 'user_based': False})
    item_MSD = cross_validate(algo, data, cv=3, verbose=False)
    plotRMSE.append(["Item-based filtering", 1, item_MSD["test_rmse"].mean()])
    plotMAE.append(["Item-based filtering", 1, item_MSD["test_mae"].mean()])

    print("-----Cosine similarity in Item based Collaborative Filtering----")
    algo = KNNBasic(sim_options={'name':'cosine', 'user_based': False})
    item_Cos = cross_validate(algo, data, cv=3, verbose=False)
    plotRMSE.append(["Item-based filtering", 2, item_Cos["test_rmse"].mean()])
    plotMAE.append(["Item-based filtering", 2, item_Cos["test_mae"].mean()])

    print("-----Pearson similarity in Item based Collaborative Filtering----")
    algo = KNNBasic(sim_options={'name':'pearson', 'user_based': False})
    item_Pearson = cross_validate(algo, data, cv=3, verbose=False)
    plotRMSE.append(["Item-based filtering", 3, item_Pearson["test_rmse"].mean()])
    plotMAE.append(["Item-based filtering", 3, item_Pearson["test_mae"].mean()])

    plotRMSE = pd.DataFrame(data=plotRMSE, columns=["Filter", "Similarity", "RMSE"])
    plotRMSE.pivot("Similarity", "Filter", "RMSE").plot(kind="bar")
    plt.title("User vs Item (RMSE)")
    plt.ylabel("RMSE")
    plt.ylim(.9, 1.1)
    plt.show()

    plotMAE = pd.DataFrame(data=plotMAE, columns=["Filter", "Similarity", "MAE"])
    plotMAE.pivot("Similarity", "Filter", "MAE").plot(kind="bar")
    plt.title("User vs Item (MAE)")
    plt.ylabel("MAE")
    plt.ylim(.7, .9)
    plt.show()

problem14()

def problem15():
    plotNeighbors = []
    i = 1
    while i < 17:
        algo = KNNBasic(k=i, sim_options={'name': 'MSD', 'user_based': True})
        user = cross_validate(algo, data, cv=3, verbose = False)
        plotNeighbors.append(["User based Collobarative Filtering", i, user["test_rmse"].mean()])
        algo = KNNBasic(k=i, sim_options={'name': 'MSD', 'user_based': False})
        item_MSD = cross_validate(algo, data, cv=3, verbose=False)
        plotNeighbors.append(["Item based Collaborative Filtering", i, item_MSD["test_rmse"].mean()])
        i+=1
    plotDF = pd.DataFrame(data=plotNeighbors, columns=["Classifier", "K", "Score"])
    plotDF.pivot("K", "Classifier", "Score").plot(kind="bar")
    plt.ylim(0.8, 1.6)
    plt.title("User/Item based collaborative filtering in terms of k-value")
    plt.ylabel("RMSE")
    plt.show()

problem15()    
