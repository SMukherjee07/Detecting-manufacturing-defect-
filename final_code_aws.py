!pip install imbalanced-learn joblib
import boto3
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
s3_bucket_name = 'gaip2024sam7'  # Bucket name without "s3://"
s3_data_key = 'Deployment/dataset/manufacturing_data.csv'  # Key (path) within the bucket
s3_model_path = 'Deployment/model/manufacturing_defect_detection_model.pkl'
data_location = f's3://gaip2024sam7/Deployment/dataset/manufacturing_defect_dataset.csv'
data = pd.read_csv(data_location)
data.head()
data.isnull().sum()  # Check for missing values
X = data.drop('DefectStatus', axis=1)
y = data['DefectStatus']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
accuracy = np.mean(y_pred == y_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
model_filename = "manufacturing_defect_detection_model.pkl"
joblib.dump(model, model_filename)

s3_client = boto3.client('s3')
s3_client.upload_file(model_filename, s3_bucket_name, s3_model_path)
print("Model saved to Amazon S3 successfully.")
s3_obj = boto3.resource('s3').Object(s3_bucket_name, s3_model_path)
model_bytes = s3_obj.get()['Body'].read()
from io import BytesIO
loaded_model = joblib.load(BytesIO(model_bytes))
y_pred_test = loaded_model.predict(X_test)
predictions_csv_filename = 'predictions.csv'
y_pred_df = pd.DataFrame({'Predicted Labels': np.where(y_pred_test == 0, 'Low Defects', 'High Defects')})
y_pred_df.to_csv(predictions_csv_filename, index=False)
s3_predictions_path = f'Deployment/predictions/{predictions_csv_filename}'
s3_client.upload_file(predictions_csv_filename, s3_bucket_name, s3_predictions_path)
print("Predictions saved to Amazon S3 successfully.")
