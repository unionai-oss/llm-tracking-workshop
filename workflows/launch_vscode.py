from pathlib import Path
import tarfile
from flytekit import task, workflow, Resources
from flytekitplugins.flyteinteractive import vscode
from flytekit.types.file import FlyteFile

CONTAINER_IMAGE = (
    "us-central1-docker.pkg.dev/uc-serverless-production/union/launch_vscode:0.0.4"
)


class custom_vscode(vscode):
    def execute(self, *args, **kwargs):
        model = kwargs["model"]

        inputs_dir = Path("/root") / "model_dir"
        inputs_dir.mkdir(exist_ok=True)

        dest_name = model.remote_source.split("/")[-1]
        print(f"Downloading file: {dest_name}")
        local_path = model.download()
        with tarfile.open(local_path, "r:gz") as tar:
            print(f"Extracting model to {inputs_dir}")
            tar.extractall(inputs_dir)

        return super().execute(*args, **kwargs)


@task(
    container_image=CONTAINER_IMAGE,
    requests=Resources(cpu="3", mem="12Gi"),
)
@custom_vscode
def start_vscode(model: FlyteFile):
    pass


@workflow
def main(model: FlyteFile):
    start_vscode(model=model)
