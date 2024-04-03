from pathlib import Path
import tarfile

from flytekitplugins.flyteinteractive import get_task_inputs
from flytekit.types.file import FlyteFile


def download_model(task_module_name: str, task_name: str, context_working_dir: str):
    inputs = get_task_inputs(
        task_module_name=task_module_name,
        task_name=task_name,
        context_working_dir=context_working_dir,
    )
    inputs_dir = Path("model_dir")
    inputs_dir.mkdir(exist_ok=True)

    for param in inputs.values():
        if not isinstance(param, FlyteFile):
            continue

        dest_name = param.remote_source.split("/")[-1]
        print(f"Downloading file: {dest_name}")
        local_path = param.download()
        if dest_name.endswith(".tar.gz"):
            with tarfile.open(local_path, "r:gz") as tar:
                tar.extractall(inputs_dir)

    print("File downloaded successfully!")
