import pandas as pd
import autokeras as ak
from autokeras import ImageClassifier, ImageRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np
import tensorflow as tf
import streamlit as st
import base64
import io
import matplotlib.pyplot as plt
import math

# Page expands to full width
st.set_page_config(page_title='AIDrugApp', page_icon='🌐', layout="wide")

# For hiding streamlit messages
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('deprecation.showfileUploaderEncoding', False)

# Create title and subtitle
html_temp = """
        <div style="background-color:teal">
        <h1 style="font-family:arial;color:white;text-align:center;">AIDrugApp</h1>
        <h4 style="font-family:arial;color:white;text-align:center;">Artificial Intelligence Based Virtual Screening Web-App for Drug Discovery</h4>
        </div>
        <br>
        """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.title("AIDrugApp v1.2.6")
st.sidebar.header("Menu")
CB = st.sidebar.checkbox("Auto-DL")

if CB:
    st.title('Auto-DL')
    st.success(
        "This module of [**AIDrugApp v1.2.5**](https://aidrugapp.streamlit.app/) helps to build the best Deep Learning model on user data."
        " It also helps to predict target data using the same deep learning algorithm.")

    expander_bar = st.expander("👉 More information")
    expander_bar.markdown("""
            * **Python libraries:** Tensorflow, AutoKeras, scikit-learn, streamlit, pandas, numpy, matplotlib
            * **Publication:** 1. Divya Karade. (2021). Custom ML Module of AIDrugApp for Molecular Identification, Descriptor Calculation, and Building ML/DL QSAR Models. [ChemRxiv Preprint](https://doi.org/10.33774/chemrxiv-2021-3f1f9).
            2. Divya Karade. (2021, March 23). AutoDL: Automated Deep Learning (Machine learning module of AIDrugApp - Artificial Intelligence Based Virtual Screening Web-App for Drug Discovery) (Version 1.0.0). [Zenodo](http://doi.org/10.5281/zenodo.4630119)
            """)

    expander_bar = st.expander("👉 How to use Auto-DL?")
    expander_bar.markdown("""**Step 1:** In the User input-side panel, select the type of algorithm ('Classification' or 'Regression') for building the DL model.""")
    expander_bar.markdown("""**Step 2:** Upload descriptor data (included with target data) for building the DL model. (*Example input file provided*)""")
    expander_bar.markdown("""**Step 3:** For developing the model, specify parameters such as 'Train-Test split percent', 'random seed number', 'maximum trial number', and 'epochs number'.""")
    expander_bar.markdown("""**Step 4:** Upload descriptor data (excluded with target data) for making target predictions. (*Example input file provided*)""")
    expander_bar.markdown("""**Step 5:** Click the "✨ PREDICT" button and the results will be displayed below to view and download.""")

    """---"""

    st.sidebar.header('⚙️ USER INPUT PANEL')
    st.sidebar.write('**1. Which Auto-DL algorithm would you like to select for building DL models?**')
    add_selectbox = st.sidebar.radio(
        "Select your algorithm",
        ("Regression", "Classification"))

    st.sidebar.write('**2. Upload data file for building deep learning models**')
    uploaded_file = st.sidebar.file_uploader("Upload input .csv file", type=["csv"])
    st.sidebar.markdown("""[Example .csv input file](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2/tree/main/Example-.csv-input-files_AutoDL)""")

    # Sidebar - Specify parameter settings
    st.sidebar.write('**3. Set Parameters**')
    split_size = st.sidebar.number_input('Train-Test split %', 0, 100, 70, 5)
    seed_number = st.sidebar.number_input('Set the random seed number', 1, 100, 42, 1)
    max_trials = st.sidebar.number_input('Set the maximum trial number', 1, 100, 15, 1)
    epochs = st.sidebar.number_input('Set the epochs number', 1, 1000, 50, 5)

    st.sidebar.write("**4. Upload data file for predictions: **")
    file_upload = st.sidebar.file_uploader("Upload .csv file", type=["csv"])
    st.sidebar.markdown("""[Example .csv input file](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2/tree/main/Example-.csv-input-files_AutoDL)""")

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        st.info("**Uploaded data for making predictions **")
        st.write('Data Dimension: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
        st.write(data.style.highlight_max(axis=0))
    else:
        st.info('Awaiting .csv file to be uploaded for making predictions')

    DA = st.sidebar.button("✨ PREDICT")

    # Load dataset
    if uploaded_file is not None:
        data_1 = pd.read_csv(uploaded_file)
        X = data_1.iloc[:, :-1]  # Using all columns except for the last column as X
        Y = data_1.iloc[:, -1]  # Selecting the last column as Y

        st.info("**Uploaded data for building DL models: **")
        st.write('Data Dimension: ' + str(data_1.shape[0]) + ' rows and ' + str(data_1.shape[1]) + ' columns.')
        st.write(data_1.style.highlight_max(axis=0))

        # Data split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=(1 - split_size / 100),
                                                            random_state=seed_number)

        st.write('**Training set**')
        Train = pd.concat([X_train, y_train], axis=1)
        st.write('Data Dimension: ' + str(Train.shape[0]) + ' rows and ' + str(Train.shape[1]) + ' columns.')
        st.write(Train)
        st.download_button('Download CSV', Train.to_csv(index=False), 'Train.csv', 'text/csv')

        st.write('**Test set**')
        Test = pd.concat([X_test, y_test], axis=1)
        st.write('Data Dimension: ' + str(Test.shape[0]) + ' rows and ' + str(Test.shape[1]) + ' columns.')
        st.write(Test)
        st.download_button('Download CSV', Test.to_csv(index=False), 'Test.csv', 'text/csv')

        if add_selectbox == 'Classification':
            if DA:
                seed(seed_number)
                tf.random.set_seed(seed_number)
                np.random.seed(seed_number)

                search = ImageClassifier(max_trials=max_trials)
                search.fit(x=X_train, y=y_train, verbose=0, epochs=epochs)

                y_pred_train = search.predict(X_train)
                y_pred_test = search.predict(X_test)

                model = search.export_model()

                st.info('**Model summary**')
                st.write(model.summary())

                st.info('**Model evaluation - Training Set**')
                loss, acc = search.evaluate(X_train, y_train, verbose=0)
                st.write('Accuracy: %.3f' % acc)
                precision = precision_score(y_train, y_pred_train)
                st.write('Precision: %f' % precision)
                recall = recall_score(y_train, y_pred_train)
                st.write('Sensitivity/Recall: %f' % recall)
                f1 = f1_score(y_train, y_pred_train)
                st.write('F1 score: %f' % f1)
                auc = roc_auc_score(y_train, y_pred_train)
                st.write('ROC AUC: %f' % auc)
                st.write("Confusion matrix")
                matrix = confusion_matrix(y_train, y_pred_train)
                st.write(matrix)

                st.info('**Model evaluation - Test Set**')
                loss, acc = search.evaluate(X_test, y_test, verbose=0)
                st.write('Accuracy: %.3f' % acc)
                precision = precision_score(y_test, y_pred_test)
                st.write('Precision: %f' % precision)
                recall1 = recall_score(y_test, y_pred_test)
                st.write('Sensitivity/Recall: %f' % recall1)
                f1 = f1_score(y_test, y_pred_test)
                st.write('F1 score: %f' % f1)
                auc = roc_auc_score(y_test, y_pred_test)
                st.write('ROC AUC: %f' % auc)
                st.write("Confusion matrix")
                matrix = confusion_matrix(y_test, y_pred_test)
                st.write(matrix)

                st.success("**Find the Predicted Results below: **")
                prediction = search.predict(data)
                data['Target_value'] = prediction
                st.write(data)
                st.download_button('Download CSV', data.to_csv(index=False), 'data.csv', 'text/csv')

                st.sidebar.warning('Prediction Created Successfully!')

        if add_selectbox == 'Regression':
            if DA:
                seed(seed_number)
                tf.random.set_seed(seed_number)
                np.random.seed(seed_number)

                search = ImageRegressor(max_trials=max_trials)
                search.fit(x=X_train, y=y_train, verbose=0, epochs=epochs)

                y_pred = search.predict(X_test)

                model = search.export_model()

                st.info('**Model summary**')
                st.write(model.summary())

                st.info('**Model evaluation - Training Set**')
                y_pred_train = search.predict(X_train)
                st.write("\n")
                st.write("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_train, y_pred_train))
                st.write("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_train, y_pred_train))
                st.write("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_train, y_pred_train)))
                st.write("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_train, y_pred_train))

                st.info('**Model evaluation - Test Set**')
                st.write("\n")
                st.write("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test, y_pred))
                st.write("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test, y_pred))
                st.write("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)))
                st.write("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test, y_pred))

                st.success("**Find the Predicted Results below: **")
                prediction = search.predict(data)
                data['Target_value'] = prediction
                st.write(data)
                st.download_button('Download CSV', data.to_csv(index=False), 'data.csv', 'text/csv')

                st.sidebar.warning('Prediction Created Successfully!')
