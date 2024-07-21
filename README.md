# DIABETES CLASSIFICATION WEB APP

This web application allows users to choose different machine learning models to predict diabetes and adjust parameters to achieve the best accuracy. Built using Streamlit, the app includes several popular classifiers such as SVM, Logistic Regression, Random Forest, K-Nearest Neighbors, Decision Tree, XGBoost, and LightGBM.

## Features

- Train and test different classifiers
- Display model performance metrics: Accuracy, Precision, and Recall
- Visualize results with Confusion Matrix, ROC Curve, and Precision-Recall Curve

## Installation and Usage

Follow these steps to set up and run the application in the google colab:

1. **Open Google Colab:** 
    Go to Google Colab and create a new notebook.

2. **Install Streamlit and Other Dependencies:**
    ```
    !pip install streamlit
    ```

3. **Create and Write the Streamlit Application:**
    ```
    %%writefile app.py
    #following by code in app.py
    ```

4. **Run the Streamlit Application and Expose the Local Server:**
    ```
    !wget -q -O - ipv4.icanhazip.com
    !npm install -g localtunnel@2.0.2
    !streamlit run app.py & npx localtunnel --port 8501

    ```

## Usage

1. **Upload Dataset:**
    - Upload a CSV file containing your diabetes data. The dataset should include the relevant medical features and an `Outcome` column for the labels.

2. **Choose Classifier:**
    - Select one of the classifiers from the sidebar (e.g., SVM, Logistic Regression, Random Forest, etc.).

3. **Set Hyperparameters:**
    - Adjust the hyperparameters for the selected classifier using the sidebar controls.

4. **Classify:**
    - Click the 'Classify' button to train the model and see the results.

5. **Visualize Metrics:**
    - Choose which metrics to plot (Confusion Matrix, ROC Curve, Precision-Recall Curve) and view them in the main panel.

6. **Show Raw Data:**
    - Optionally, display the raw data by checking the 'Show Raw Data' checkbox.


## Note
- After running the commands, you will see a URL provided by LocalTunnel in the output. Click on this URL to access the web app, and paste the IP address shown in the output to ensure it runs properly.
- The interface allows you to upload the diabetes.csv file, select a classifier, adjust hyperparameters, and visualize the model’s performance metrics.
