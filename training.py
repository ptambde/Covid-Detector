import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

def data_split(data):
    shuffled= np.random.permutation(len(data))
    train_indices= shuffled[:]
    return data.iloc[train_indices]

if __name__ == "__main__":
    covid_df= pd.read_excel(r'D:\Python\Machine Learning\Covid Detector\Covid_Dataset_4.xlsx')
    TRAIN = data_split(covid_df)
    X_TRAIN= TRAIN[['BodyTemp', 'BodyPain', 'DryCough', 'DiffBreathing', 'Age']].to_numpy()
    Y_TRAIN= TRAIN[['InfectionProb']].to_numpy().reshape(len(TRAIN.index),)
    model= LogisticRegression()
    model.fit(X_TRAIN, Y_TRAIN)
    file= open(r'D:\Python\Machine Learning\Covid Detector\model.pkl', 'wb')
    pickle.dump(model, file)
    file.close()        