import streamlit as st
import pandas as pd
import numpy as np
pip install --upgrade scikit-learn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, precision_score, recall_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import xgboost as xgb
from lightgbm import LGBMClassifier

def main():
    st.title("Diabetes Classification Web App")
    st.subheader("Determine if you have diabetes based on medical data.")
    st.sidebar.title("Train your Own Model using Classification Techniques mentioned below:")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("diabetes.csv")
        st.write("Columns in the dataset:", data.columns)
        return data

    @st.cache_data(persist=True)
    def split(df):
        y = df['Outcome']
        x = df.drop(columns=['Outcome'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        return x_train_scaled, x_test_scaled, y_train, y_test

    def plot_metrics(metrics_list, y_test, y_pred, model, x_test_scaled):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            if hasattr(model, "decision_function"):
                y_score = model.decision_function(x_test_scaled)
            else:
                y_score = model.predict_proba(x_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            fig, ax = plt.subplots()
            roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
            roc_disp.plot(ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            if hasattr(model, "decision_function"):
                y_score = model.decision_function(x_test_scaled)
            else:
                y_score = model.predict_proba(x_test_scaled)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            fig, ax = plt.subplots()
            pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            pr_disp.plot(ax=ax)
            st.pyplot(fig)

    df = load_data()
    x_train_scaled, x_test_scaled, y_train, y_test = split(df)
    class_names = [0, 1]
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier:", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "K-Nearest Neighbors (KNN)", "Decision Tree", "XGBoost", "LightGBM"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ('scale', 'auto'), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train_scaled, y_train)
            accuracy = model.score(x_test_scaled, y_test)
            y_pred = model.predict(x_test_scaled)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='binary').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='binary').round(2))
            plot_metrics(metrics, y_test, y_pred, model, x_test_scaled)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum Number of Iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train_scaled, y_train)
            accuracy = model.score(x_test_scaled, y_test)
            y_pred = model.predict(x_test_scaled)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='binary').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='binary').round(2))
            plot_metrics(metrics, y_test, y_pred, model, x_test_scaled)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of Trees in Forest", 100, 500, step=10, key='n_estimators')
        max_depth = st.sidebar.slider("Maximum Depth of the Tree", 1, 20, key='max_depth')
        bootstrap_str = st.sidebar.radio("Bootstrap Samples when Building Trees?", ('True', 'False'), key='bootstrap')
        bootstrap = bootstrap_str == 'True'  # Convert string to boolean

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train_scaled, y_train)
            accuracy = model.score(x_test_scaled, y_test)
            y_pred = model.predict(x_test_scaled)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='binary').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='binary').round(2))
            plot_metrics(metrics, y_test, y_pred, model, x_test_scaled)

    if classifier == 'K-Nearest Neighbors (KNN)':
        st.sidebar.subheader("Model Hyperparameters")
        n_neighbors = st.sidebar.number_input("Number of Neighbors", 1, 20, step=1, key='n_neighbors')
        weights = st.sidebar.radio("Weights", ("uniform", "distance"), key='weights')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader("K-Nearest Neighbors (KNN) Results")
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            model.fit(x_train_scaled, y_train)
            accuracy = model.score(x_test_scaled, y_test)
            y_pred = model.predict(x_test_scaled)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='binary').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='binary').round(2))
            plot_metrics(metrics, y_test, y_pred, model, x_test_scaled)

    if classifier == 'Decision Tree':
        st.sidebar.subheader("Model Hyperparameters")
        max_depth = st.sidebar.slider("Maximum Depth of the Tree", 1, 20, key='max_depth_dt')
        criterion = st.sidebar.radio("Criterion", ("gini", "entropy"), key='criterion')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader("Decision Tree Results")
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
            model.fit(x_train_scaled, y_train)
            accuracy = model.score(x_test_scaled, y_test)
            y_pred = model.predict(x_test_scaled)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='binary').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='binary').round(2))
            plot_metrics(metrics, y_test, y_pred, model, x_test_scaled)

    if classifier == 'XGBoost':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of Estimators", 100, 1000, step=10, key='n_estimators_xgb')
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, key='learning_rate')
        max_depth = st.sidebar.slider("Maximum Depth of the Tree", 1, 20, key='max_depth_xgb')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader("XGBoost Results")
            model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, use_label_encoder=False, eval_metric='logloss')
            model.fit(x_train_scaled, y_train)
            accuracy = model.score(x_test_scaled, y_test)
            y_pred = model.predict(x_test_scaled)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='binary').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='binary').round(2))
            plot_metrics(metrics, y_test, y_pred, model, x_test_scaled)

    if classifier == 'LightGBM':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of Estimators", 100, 1000, step=10, key='n_estimators_lgbm')
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, key='learning_rate_lgbm')
        max_depth = st.sidebar.slider("Maximum Depth of the Tree", 1, 20, key='max_depth_lgbm')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader("LightGBM Results")
            model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            model.fit(x_train_scaled, y_train)
            accuracy = model.score(x_test_scaled, y_test)
            y_pred = model.predict(x_test_scaled)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='binary').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='binary').round(2))
            plot_metrics(metrics, y_test, y_pred, model, x_test_scaled)

    if st.sidebar.checkbox("Show Raw Data", False):
        st.subheader("Diabetes Data Set")
        st.write(df)

if __name__ == '__main__':
    main()
