import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import replace
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.title("Machine Learning Classification Project")

st.write(""" Dataset: """)
dataset_name = st.sidebar.selectbox("Select Dataset", ("Breast Cancer",
                                                       "Diabetes", "Wine Quality", "Iris"))

st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN",
                                                             "Naive Bayes", "SVM", "Random Forrest", "Kernel SVM"))


def load_dataset(dataset_name):
    data = None

    if dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()

    elif dataset_name == 'Wine Quality':
        data = datasets.load_wine()
    elif dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Diabetes':
        data = datasets.load_diabetes()

    X = data.data
    y = data.target

    return X, y


X, y = load_dataset(dataset_name)
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


def add_classifier(cl_name):
    params = dict()
    if cl_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K

    elif cl_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif cl_name == "Naive Bayes":
        GaussianNB = st.sidebar.write("GaussianNB")
        params["GaussianNB"] = GaussianNB
    elif cl_name == "Kernel SVM":
        rbf = st.sidebar.write("rbf")
        params["rbf"] = rbf
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params


params = add_classifier(classifier_name)


def implement_classifier(cl_name, params):
    cl = None
    if cl_name == 'Naive Bayes':
        cl = GaussianNB()
    elif cl_name == 'SVM':
        cl = SVC(C=params['C'])
    elif cl_name == 'KNN':
        cl = KNeighborsClassifier(n_neighbors=params['K'])
    elif cl_name == 'Kernel SVM':
        cl = SVC()
    else:
        cl = RandomForestClassifier(n_estimators=params['n_estimators'],
                                    max_depth=params['max_depth'], random_state=1234)
    return cl


cl = implement_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)
cl.fit(X_train, y_train)
y_pred = cl.predict(X_test)

acc = accuracy_score(y_test, y_pred)
confusionmatrix = confusion_matrix(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
st.write("Confusion Matrix:")
st.table(data=confusionmatrix)


pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()


st.pyplot(fig)
