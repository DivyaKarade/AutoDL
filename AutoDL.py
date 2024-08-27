import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
)
from autokeras import StructuredDataClassifier, StructuredDataRegressor
import matplotlib.pyplot as plt

st.title("AutoKeras Tabular Data Modeling")

# Sidebar options
st.sidebar.title("Options")
add_selectbox = st.sidebar.selectbox("Choose a task", ("Classification", "Regression"))

# Upload dataset
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Data preprocessing
    target = st.sidebar.selectbox("Select the target column", data.columns)
    X = data.drop(columns=[target])
    y = data[target]

    test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
    random_state = st.sidebar.number_input("Random State (for reproducibility)", value=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.write("Training set size:", X_train.shape)
    st.write("Test set size:", X_test.shape)

    # Model training
    if add_selectbox == 'Classification':
        st.subheader("AutoKeras Structured Data Classification")

        max_trials = st.sidebar.slider("Max Trials", 1, 100, 10)
        epochs = st.sidebar.slider("Epochs", 1, 100, 10)
        seed_number = st.sidebar.number_input("Seed number", value=42)

        tf.random.set_seed(seed_number)
        np.random.seed(seed_number)

        search = StructuredDataClassifier(max_trials=max_trials)
        search.fit(x=X_train, y=y_train, verbose=0, epochs=epochs)

        y_pred_train = search.predict(X_train)
        y_pred_test = search.predict(X_test)

        model = search.export_model()

        st.info('**Model summary**')
        st.write(model.summary())

        st.info('**Model evaluation - Training Set**')
        loss, acc = search.evaluate(X_train, y_train, verbose=0)
        st.write('Accuracy: %.3f' % acc)
        precision = precision_score(y_train, y_pred_train, average="weighted")
        st.write('Precision: %f' % precision)
        recall = recall_score(y_train, y_pred_train, average="weighted")
        st.write('Sensitivity/Recall: %f' % recall)
        f1 = f1_score(y_train, y_pred_train, average="weighted")
        st.write('F1 score: %f' % f1)
        auc = roc_auc_score(y_train, search.predict_proba(X_train), multi_class='ovo')
        st.write('ROC AUC: %f' % auc)
        st.write("Confusion matrix")
        matrix = confusion_matrix(y_train, y_pred_train)
        st.write(matrix)

        st.info('**Model evaluation - Test Set**')
        loss, acc = search.evaluate(X_test, y_test, verbose=0)
        st.write('Accuracy: %.3f' % acc)
        precision = precision_score(y_test, y_pred_test, average="weighted")
        st.write('Precision: %f' % precision)
        recall1 = recall_score(y_test, y_pred_test, average="weighted")
        st.write('Sensitivity/Recall: %f' % recall1)
        f1 = f1_score(y_test, y_pred_test, average="weighted")
        st.write('F1 score: %f' % f1)
        auc = roc_auc_score(y_test, search.predict_proba(X_test), multi_class='ovo')
        st.write('ROC AUC: %f' % auc)
        st.write("Confusion matrix")
        matrix = confusion_matrix(y_test, y_pred_test)
        st.write(matrix)

    elif add_selectbox == 'Regression':
        st.subheader("AutoKeras Structured Data Regression")

        max_trials = st.sidebar.slider("Max Trials", 1, 100, 10)
        epochs = st.sidebar.slider("Epochs", 1, 100, 10)
        seed_number = st.sidebar.number_input("Seed number", value=42)

        tf.random.set_seed(seed_number)
        np.random.seed(seed_number)

        search = StructuredDataRegressor(max_trials=max_trials)
        search.fit(x=X_train, y=y_train, verbose=0, epochs=epochs)

        y_pred_train = search.predict(X_train)
        y_pred_test = search.predict(X_test)

        model = search.export_model()

        st.info('**Model summary**')
        st.write(model.summary())

        st.info('**Model evaluation - Training Set**')
        loss = search.evaluate(X_train, y_train, verbose=0)
        st.write('Mean Squared Error: %.3f' % loss)

        st.info('**Model evaluation - Test Set**')
        loss = search.evaluate(X_test, y_test, verbose=0)
        st.write('Mean Squared Error: %.3f' % loss)

        # Error Distribution
        st.info('**Error Distribution - Test Set**')
        Error = pd.DataFrame(y_test)
        Error['Prediction'] = y_pred_test
        Error['Error'] = Error['Prediction'] - Error.iloc[:, 0]

        st.write(Error)

        st.info('**Plot of Prediction vs Real Value**')
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
        plt.xlabel('Real Value')
        plt.ylabel('Prediction')
        plt.title('Prediction vs Real Value')
        st.pyplot(plt)

        st.info('**Plot of Error Distribution**')
        plt.figure(figsize=(10, 6))
        plt.hist(Error['Error'], bins=25)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        st.pyplot(plt)

else:
    st.info('Awaiting .csv file to be uploaded. A sample dataset is available in the sidebar.')
