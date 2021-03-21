# pip install -U keras-tuner
# pip install autokeras

import pandas as pd
import numpy as np
from autokeras import StructuredDataClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, roc_auc_score, f1_score, \
    cohen_kappa_score, confusion_matrix
from sklearn import metrics
from numpy.random import seed
import tensorflow as tf
import streamlit as st
import base64
import io
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

add_selectbox = st.sidebar.radio(
    "Select ML tool",
    ("Auto-Multi-ML", "Auto-DL"))

if add_selectbox == 'Auto-DL':
    st.title('Auto-DL')
    st.success(
        "This module of **AIDrugApp v1.2** helps to apply and compare multiple machine learning models at once to select the best "
        "performing machine learning algorithm on users data. It also helps to predict target data based on user specific machine learning models.")

    # expander_bar = st.beta_expander("👉 More information")
    expander_bar = st.beta_expander("👉 How to use Auto-Multi-ML?")
    expander_bar.markdown("""
                    **Step 1:** On the "User Input Panel" first select AutoML algorithm for comparing multiple machine learning models
                    """)
    expander_bar.markdown("""
                    **Step 2:** Upload data (included with target data) for building multiple ML models (*Example input file given*)
                    """)
    expander_bar.markdown("""
                    **Step 3:** Select the checkbox by clicking for exploratory data analysis and/or for interpreting and comparing multiple ML models
                    """)
    expander_bar.markdown("""
                    **Step 4:** Pick your ML algorithm (preferably best perfoming ML model for best results) for building ML model on your data after comparing and interpreting results of mutiple ML models
                    """)
    expander_bar.markdown("""
                    **Step 5:** Upload data (excluded with target data) based on feature engineered model data for making target predictions by applying chosen built ML model (*Example input file given*)
                    """)
    expander_bar.markdown("""
                    **Step 6:** Click the "Predict" button and the results will be displayed below
                    """)

    """---"""

    st.sidebar.subheader('⚙ User Input Panel')
    st.sidebar.write('**1. Which Auto-DL algorithm would you like to select for predicting model performance?**')
    add_selectbox = st.sidebar.radio(
        "Select your algorithm",
        ("Regression", "Classification"))

    st.sidebar.write('**2. Upload data file for building deep learning models**')
    uploaded_file = st.sidebar.file_uploader("Upload input .csv file", type=["csv"])
    st.sidebar.markdown("""[Example .csv input files](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2)
                                    """)

    # Sidebar - Specify parameter settings
    st.sidebar.write('**3. Set Parameters**')
    split_size = st.sidebar.slider('Train-Test split %', 0, 100, 70, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)
    max_trials = st.sidebar.slider('Set the maximum trail number', 1, 100, 15, 1)
    epochs = st.sidebar.slider('Set the epochs number', 1, 1000, 50, 5)

    st.sidebar.write("**4. Upload data file for predictions: **")
    file_upload = st.sidebar.file_uploader("Upload .csv file", type=["csv"])
    st.sidebar.markdown("""[Example .csv input batch file](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2)
                                                                            """)
    if file_upload is not None:
        data = pd.read_csv(file_upload)
        # data1 = data.dropna()
        features = data.iloc[:, 0:100]
        X = features

        st.info("**Uploaded data for prediction: **")
        st.write('Data Dimension: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
        st.write(data.style.highlight_max(axis=0))

    else:
        st.info('Awaiting .csv file to be uploaded for making predictions')

    DA = st.sidebar.button("PREDICT")

    # Load dataset
    if uploaded_file is not None:
        data_1 = pd.read_csv(uploaded_file)
        # data_1 = data_1.loc[:10000]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        X = data_1.iloc[:, :-1]  # Using all column except for the last column as X
        Y = data_1.iloc[:, -1]  # Selecting the last column as Y
        # labels = data_1['Activity_value']
        # features = data_1.iloc[:, 0:8]
        # X = features
        # y = np.ravel(labels)
        st.info("**Uploaded data for building multiple ML models: **")
        st.write('Data Dimension: ' + str(data_1.shape[0]) + ' rows and ' + str(data_1.shape[1]) + ' columns.')
        st.write(data_1.style.highlight_max(axis=0))

        # Data split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=(1 - split_size / 100), random_state=seed_number)

        st.write('**Training set**')
        df1 = pd.DataFrame(X_train)
        # data_1 = data_1.loc[:1000]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION

        df2 = pd.DataFrame(y_train)
        # data_1 = data_1.loc[:1000]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION

        frames1 = [df1, df2]
        Train = pd.concat(frames1, axis=1)
        st.write('Data Dimension: ' + str(Train.shape[0]) + ' rows and ' + str(Train.shape[1]) + ' columns.')
        st.write(Train)


        # Download predicted data
        def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="Train.csv">⏬ Download CSV File</a>'
            return href


        st.markdown(filedownload(Train), unsafe_allow_html=True)

        st.write('**Test set**')
        df3 = pd.DataFrame(X_test)
        df4 = pd.DataFrame(y_test)
        frames2 = [df3, df4]
        Test = pd.concat(frames2, axis=1)
        st.write('Data Dimension: ' + str(Test.shape[0]) + ' rows and ' + str(Test.shape[1]) + ' columns.')
        st.write(Test)


        # Download predicted data
        def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="Test.csv">⏬ Download CSV File</a>'
            return href


        st.markdown(filedownload(Test), unsafe_allow_html=True)

        if add_selectbox == 'Classification':
            if DA:

                seed(2)
                tf.random.set_seed(2)
                # set the seeds for reproducible results with TF (wont work with GPU, only CPU)
                np.random.seed(2)

                session_conf = tf.compat.v1.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1)

                # Force Tensorflow to use a single thread
                sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

                tf.compat.v1.keras.backend.set_session(sess)

                # X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=2)
                #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_size,random_state=seed_number)


                # define the search
                search = StructuredDataClassifier(max_trials=max_trials)
                # perform the search
                search.fit(x=X_train, y=y_train, verbose=0, epochs=epochs)

                y_pred_train = search.predict(X_train)
                y_pred_test = search.predict(X_test)

                # get the best performing model
                model = search.export_model()

                # Model summary
                s = io.StringIO()
                model.summary(print_fn=lambda x: s.write(x + '\n'))
                model_summary = s.getvalue()
                s.close()

                print("The model summary is:\n\n{}".format(model_summary))
                st.subheader('Model summary')
                st.write("Model parameters")
                plt.text(0.1, 0.1, model_summary)
                plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
                plt.grid(False)
                st.pyplot()

                # prediction = pd.DataFrame(prediction, columns=['predictions']).to_csv('cas_2_prediction.csv')

                # Training set
                st.info('Training Set')
                # evaluate the model
                loss, acc = search.evaluate(X_train, y_train, verbose=0)
                st.write('Accuracy: %.3f' % acc)
                # precision tp / (tp + fp)
                precision = precision_score(y_train, y_pred_train)
                st.write('Precision: %f' % precision)
                # recall: tp / (tp + fn)
                recall = recall_score(y_train, y_pred_train)
                st.write('Sensitivity/Recall: %f' % recall)
                # f1: 2 tp / (2 tp + fp + fn)
                f1 = f1_score(y_train, y_pred_train)
                st.write('F1 score: %f' % f1)
                # ROC AUC
                auc = roc_auc_score(y_train, y_pred_train)
                st.write('ROC AUC: %f' % auc)
                # confusion matrix
                matrix = confusion_matrix(y_train, y_pred_train)
                st.write(matrix)

                # Test set
                st.info('Test Set')
                loss, acc = search.evaluate(X_test, y_test, verbose=0)
                st.write('Accuracy: %.3f' % acc)
                # precision tp / (tp + fp)
                precision = precision_score(y_test, y_pred_test)
                st.write('Precision: %f' % precision)
                # recall: tp / (tp + fn) [Sensitivity/Recall]
                recall1 = recall_score(y_test, y_pred_test)
                st.write('Sensitivity/Recall: %f' % recall)
                # f1: 2 tp / (2 tp + fp + fn)
                f1 = f1_score(y_test, y_pred_test)
                st.write('F1 score: %f' % f1)
                # ROC AUC
                auc = roc_auc_score(y_test, y_pred_test)
                st.write('ROC AUC: %f' % auc)
                # confusion matrix
                matrix = confusion_matrix(y_test, y_pred_test)
                st.write(matrix)

                st.info("**Find the Predicted Results below: **")
                prediction = search.predict(data)
                data['Target_value'] = prediction
                st.write(data)


                # Download predicted data
                def filedownload(df):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
                    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">⏬ Download CSV File</a>'
                    return href


                st.markdown(filedownload(data), unsafe_allow_html=True)

                st.sidebar.warning('Prediction Created Sucessfully!')
