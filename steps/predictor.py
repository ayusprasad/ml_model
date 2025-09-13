import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def predictor(
    service:MLFlowDeploymentService,
    input_data:str,
) -> np.ndarray:
    
    service.start(timeout=10)
    data=json.loads(input_data)
    
    data.pop("columns",None)
    data.pop("index",None)
    
    expected_columns=[
    "No of Bedrooms",
    "No of Bathrooms",
    "Flat Area (in Sqft)",
    "Lot Area (in Sqft)",
    "No of Floors",
    "Overall Grade",
    "Area of the House from Basement (in Sqft)",
    "Basement Area (in Sqft)",
    "Age of House (in Years)",
    "Renovated Year",
    "Latitude",
    "Longitude",
    "Living Area after Renovation (in Sqft)",
    "Lot Area after Renovation (in Sqft)",
    ]
    
    
    df=pd.DataFrame(data["data"],columns=expected_columns)
    
    json_list=json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array=np.array(json_list)
    prediction=service.predict(data_array)
    return prediction

if __name__=="__main__":
    print("predictor  is runnnig")
