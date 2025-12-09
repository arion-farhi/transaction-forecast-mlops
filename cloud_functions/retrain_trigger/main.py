
import functions_framework
from google.cloud import aiplatform

PROJECT_ID = "transaction-forecast-mlops"
REGION = "us-central1"
PIPELINE_ROOT = "gs://transaction-forecast-data/pipeline_root"
TEMPLATE_PATH = "gs://transaction-forecast-data/pipeline/transaction_forecast_pipeline.json"

@functions_framework.http
def trigger_retrain(request):
    """Trigger pipeline retraining based on drift detection or scheduled call."""
    
    request_json = request.get_json(silent=True)
    
    trigger_reason = "manual"
    random_state = 42
    
    if request_json:
        trigger_reason = request_json.get("trigger_reason", "manual")
        random_state = request_json.get("random_state", 42)
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    job = aiplatform.PipelineJob(
        display_name=f"triggered-{trigger_reason}",
        template_path=TEMPLATE_PATH,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "bucket_name": "transaction-forecast-data",
            "project_id": PROJECT_ID,
            "region": REGION,
            "model_name": "transaction-forecast-xgboost",
            "initial_threshold": 10.0,
            "max_threshold": 15.0,
            "min_improvement": 0.5,
            "random_state": random_state
        }
    )
    
    job.submit()
    
    return {
        "status": "submitted",
        "trigger_reason": trigger_reason,
        "job_name": job.display_name
    }
