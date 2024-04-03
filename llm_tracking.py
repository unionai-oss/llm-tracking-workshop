"""Runs ML workflow with wandb integration.

1. Download dataset
2. Download model weights
3. Train model and upload data to wandb
4. Configure artifact metadata in wandb
5. Run a second workflow with another model and compare models

"""

from typing import Optional
from pathlib import Path
import tarfile
import os
from flytekit import task, workflow, current_context, ImageSpec, Secret, Resources
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from flytekit.core.context_manager import FlyteContextManager
from flytekit.models.core.identifier import WorkflowExecutionIdentifier

from unionai._config import _get_organization, _get_config_obj

state = FlyteContextManager.current_context().execution_state
endpoint, org = "", None
# Only runs locally
if state and state.mode is None:
    platform = _get_config_obj().platform
    endpoint = platform.endpoint
    org = _get_organization(platform)


def _get_execution_url(
    endpoint: str, org: Optional[str], execution_id: WorkflowExecutionIdentifier
) -> str:
    org_prefix = "console" if org is None else f"org/{org}"
    return (
        f"https://{endpoint}/{org_prefix}/"
        f"projects/{execution_id.project}/"
        f"domains/{execution_id.domain}/"
        f"executions/{execution_id.name}"
    )


wandb_image = ImageSpec(cuda="11.8", requirements="requirements_wandb.txt")


@task(container_image=wandb_image, cache=True, cache_version="v5")
def download_dataset() -> FlyteDirectory:
    from datasets import load_dataset

    working_dir = Path(current_context().working_directory)
    dataset_cache_dir = working_dir / "dataset_cache"

    load_dataset("imdb", cache_dir=dataset_cache_dir)

    return dataset_cache_dir


@task(container_image=wandb_image, cache=True, cache_version="v5")
def download_model(model: str) -> FlyteDirectory:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    working_dir = Path(current_context().working_directory)
    model_cache_dir = working_dir / "model_cache"

    AutoTokenizer.from_pretrained(model, cache_dir=model_cache_dir)
    AutoModelForSequenceClassification.from_pretrained(model, cache_dir=model_cache_dir)
    return model_cache_dir


@task(
    container_image=wandb_image,
    secret_requests=[Secret(key="wandb_api_key")],
    requests=Resources(cpu="2", mem="12Gi", gpu="1"),
)
def train_model(
    model: str,
    wandb_project: str,
    model_cache_dir: FlyteDirectory,
    dataset_cache_dir: FlyteDirectory,
    endpoint: str,
    org: Optional[str],
) -> tuple[str, FlyteFile]:
    from datasets import load_dataset
    import numpy as np

    import wandb
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        pipeline,
    )

    ctx = current_context()
    secrets = ctx.secrets
    wandb.login(key=secrets.get(key="wandb_api_key"))

    working_dir = Path(ctx.working_directory)
    train_dir = working_dir / "models"

    dataset = load_dataset("imdb", cache_dir=dataset_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=model_cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        cache_dir=model_cache_dir,
    )

    def tokenizer_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Use a small subset such that finetuning completes
    small_train_dataset = (
        dataset["train"].shuffle(seed=42).select(range(500)).map(tokenizer_function)
    )
    small_eval_dataset = (
        dataset["test"].shuffle(seed=42).select(range(100)).map(tokenizer_function)
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": np.mean(predictions == labels)}

    execution_id = ctx.execution_id
    url = _get_execution_url(endpoint, org, execution_id)

    os.environ["WANDB_API_KEY"] = ctx.secrets.get(key="wandb_api_key")
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "end"

    run = wandb.init(project=wandb_project, tags=[model], save_code=True)
    run.notes = f"[See Execution]({url})"

    training_args = TrainingArguments(
        output_dir=train_dir,
        evaluation_strategy="epoch",
        report_to="wandb",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    wandb.finish()

    wandb_url = run.get_url()
    inference_path = working_dir / "inference_pipe"
    inference_pipe = pipeline("text-classification", tokenizer=tokenizer, model=model)
    inference_pipe.save_pretrained(inference_path)

    inference_path_compressed = working_dir / "inference_pipe.tar.gz"
    with tarfile.open(inference_path_compressed, "w:gz") as tar:
        tar.add(inference_path, arcname="")

    return wandb_url, inference_path_compressed


@workflow
def main(
    model: str,
    wandb_project: str,
    org: Optional[str] = org,
    endpoint: str = endpoint,
) -> str:
    dataset_cache_dir = download_dataset()
    model_cache_dir = download_model(model=model)
    url, path = train_model(
        model=model,
        wandb_project=wandb_project,
        model_cache_dir=model_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
        endpoint=endpoint,
        org=org,
    )
    return url
