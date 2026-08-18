import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
    classification_report
)


# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Online Shoppers Purchase Intention Prediction")

st.write(
    """
    This application compares different machine learning classification
    models for predicting whether an online shopping session will result
    in a purchase.
    """
)


# ----------------------------
# Available models
# ----------------------------
model_paths = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib"
}


# ----------------------------
# Model selection
# ----------------------------
selected_model = st.selectbox(
    "Select a Machine Learning Model",
    list(model_paths.keys())
)


# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"]
)


if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.dataframe(test_data.head())

    st.write("Number of test instances:", len(test_data))

    # Check for target column
    if "Revenue" not in test_data.columns:

        st.error(
            "The uploaded test dataset must contain the 'Revenue' column "
            "to calculate evaluation metrics."
        )

    else:

        # Separate features and target
        X_test = test_data.drop("Revenue", axis=1)
        y_test = test_data["Revenue"]

        # Convert target to 0/1 if necessary
        if y_test.dtype == "object":

            y_test = (
                y_test.astype(str)
                .str.lower()
                .map({
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0
                })
            )

        else:
            y_test = y_test.astype(int)


        # ----------------------------
        # Load selected model
        # ----------------------------
        model = joblib.load(model_paths[selected_model])


        # ----------------------------
        # Predictions
        # ----------------------------
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]


        # ----------------------------
        # Evaluation metrics
        # ----------------------------
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )
        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )


        # ----------------------------
        # Display metrics
        # ----------------------------
        st.subheader(f"Evaluation Results - {selected_model}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")


        # ----------------------------
        # Confusion matrix
        # ----------------------------
        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No Purchase", "Purchase"],
            ax=ax
        )

        st.pyplot(fig)


        # ----------------------------
        # Classification report
        # ----------------------------
        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df.round(4))