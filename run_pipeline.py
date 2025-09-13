import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
def main():
    run=ml_pipeline()
    trained_model=run["model_building_step"]
    print(f"trained model type : {type(trained_model)}")
    
    print(f"url == {get_tracking_uri}")
    

if __name__=="__main__":
    main()
    