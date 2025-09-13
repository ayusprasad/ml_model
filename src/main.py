import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from model_evaluator import ModelEvaluator, RegressionEvaluationStrategy
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib,json
import os
from model_evaluator import ModelEvaluator, RegressionEvaluationStrategy

data=pd.read_csv(r"C:\Users\ayush\OneDrive\Desktop\new model\data\raw\house_price.csv")
x=data.drop(columns=["Sale Price","ID","Date House was Sold"])
y=data["Sale Price"]



numeric_features=x.select_dtypes(include=["int64","float64"]).columns
categorical_features=x.select_dtypes(include=["object"]).columns

preprocessor=ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numeric_features),
        ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)
    ]
)
model=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("regressor",LinearRegression())
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# model=LinearRegression()
model.fit(x_train,y_train)
model_evaluator=ModelEvaluator(RegressionEvaluationStrategy())
evaluation_metrics=model_evaluator.evaluate(model,x_test,y_test)
print("ecaluation metrics: ",evaluation_metrics)

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# Save model
joblib.dump(model, "artifacts/linear_regression_model.pkl")
print("✅ Model saved at artifacts/linear_regression_model.pkl")
# Save evaluation metrics
with open("artifacts/evaluation_metrics.json", "w") as f:
    json.dump(evaluation_metrics, f, indent=4)
print("✅ Evaluation metrics saved at artifacts/evaluation_metrics.json")


joblib.dump(model,"artifacts/linear_regression_model.pkl")
with open("artifacts/evaluation_metrics.json","w") as f:
    json.dump(evaluation_metrics,f,indent=4)
    

feature_names=(numeric_features.tolist()
               + model.named_steps["preprocessor"]
               .named_transformers_["cat"]
               .get_feature_names_out(categorical_features)
               .tolist()
)           


coef_df=pd.DataFrame({
    "Feature":feature_names,
    "Coefficient":model.named_steps["regressor"].coef_
})

print(coef_df.sort_values(by="Coefficient",ascending=False))

y_pred=model.predict(x_test)
print("sample predictions: ",y_pred[:5])



print("*************")
evaluator=ModelEvaluator(RegressionEvaluationStrategy())
metrics=evaluator.evaluate(model,x_test,y_test)
print("Evaluation Metrics:",metrics)