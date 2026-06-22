"""
Kubeflow Pipeline: SDG → Fine-Tune → Evaluate

Components are imported directly from the existing scripts.

Run locally:
    python kfp_pipeline.py
"""

import kfp.dsl as dsl
from kfp import local

from sdg_pipeline.pipeline import sdg_component
from fine_tune import fine_tune_component
from evaluate import evaluate_component

local.init(runner=local.SubprocessRunner())


@dsl.pipeline(
    name="IT Help Desk — SDG + Fine-Tune + Evaluate",
    description="Fetch traces → judge → generate 3x synthetic data → LoRA fine-tune → evaluate",
)
def ithelpdesk_pipeline(
    mlflow_tracking_uri: str,
    mlflow_workspace: str,
    mlflow_tracking_token: str,
    experiment_name: str,
    model_url: str,
    api_key: str = "no-key-required",
    base_model: str = "Qwen/Qwen2-0.5B-Instruct",
):
    sdg_task = sdg_component(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_workspace=mlflow_workspace,
        mlflow_tracking_token=mlflow_tracking_token,
        experiment_name=experiment_name,
        model_url=model_url,
        api_key=api_key,
    )

    ft_task = fine_tune_component(base_model=base_model)
    ft_task.after(sdg_task)

    eval_task = evaluate_component(base_model=base_model)
    eval_task.after(ft_task)


if __name__ == "__main__":
    ithelpdesk_pipeline(
        mlflow_tracking_uri="<MLFLOW_TRACKING_URI>",
        mlflow_workspace="<MLFLOW_WORKSPACE>",
        mlflow_tracking_token="<MLFLOW_TRACKING_TOKEN>",
        experiment_name="<EXPERIMENT_NAME>",
        model_url="<MODEL_URL>",
    )
