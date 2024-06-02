import mlflow
from mlflow.models import infer_signature

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def modelresults(target, predictions):
    mae = mean_absolute_error(target, predictions)
    r2 = r2_score(target, predictions)
    
    print('Mean absolute error on model is {:.4f}'.format(mae))
    print('')
    print('The r2 score on model is {:.4f}'.format(r2))
    
    return mae, r2


def log_mlflow(params, mae, mse, r2, fig, X_train, model, run_name='new_model', description='New model'):
    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)
        mlflow.log_figure(fig, "figure.png")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", description)

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="SKU_forecast",
            signature=signature,
            input_example=X_train,
            registered_model_name=run_name,
        )