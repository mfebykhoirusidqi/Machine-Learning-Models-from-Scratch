import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionScratch
from logistic_regression import LogisticRegressionScratch
from kmeans import KMeansScratch

st.set_page_config(page_title="ML From Scratch", page_icon="🤖", layout="wide")
st.title("🤖 Machine Learning Models from Scratch")
st.caption("Built with pure Python + NumPy | by M Feby Khoiru Sidqi")

model_type = st.sidebar.selectbox("Select Model:", ["Linear Regression", "Logistic Regression", "K-Means Clustering"])

# Linear Regression
if model_type == "Linear Regression":
    st.header("📈 Linear Regression (From Scratch)")
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 3 * X.squeeze() + 5 + np.random.randn(50)
    model = LinearRegressionScratch(lr=0.01, epochs=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Actual")
    ax.plot(X, y_pred, color="red", label="Predicted")
    ax.set_title("Linear Regression Fit")
    ax.legend()
    st.pyplot(fig)

# Logistic Regression
elif model_type == "Logistic Regression":
    st.header("📊 Logistic Regression (From Scratch)")
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegressionScratch(lr=0.1, epochs=2000)
    model.fit(X, y)
    y_pred = model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="coolwarm")
    ax.set_title("Logistic Regression Decision Boundary")
    st.pyplot(fig)

# K-Means Clustering
elif model_type == "K-Means Clustering":
    st.header("🔵 K-Means Clustering (From Scratch)")
    X = np.random.rand(100, 2)
    model = KMeansScratch(k=3)
    model.fit(X)
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis")
    ax.scatter(model.centroids[:, 0], model.centroids[:, 1], c="red", marker="X", s=200)
    ax.set_title("K-Means Cluster Visualization")
    st.pyplot(fig)
