import signal
import multiprocessing
import os
import shutil
from pathlib import Path
import tarfile
from flytekit import task, workflow, Resources
from flytekitplugins.flyteinteractive import vscode
from flytekitplugins.flyteinteractive.vscode_lib.decorator import (
    TASK_FUNCTION_SOURCE_PATH,
    download_vscode,
    prepare_interactive_python,
    prepare_resume_task_python,
    prepare_launch_json,
    execute_command,
    resume_task_handler,
    exit_handler,
    VSCODE_TYPE_VALUE,
)
from flytekit.types.file import FlyteFile
import flytekit
import inspect

from flytekit.core.context_manager import FlyteContextManager

CONTAINER_IMAGE = (
    "us-central1-docker.pkg.dev/uc-serverless-production/union/launch_vscode:0.0.7"
)


class custom_vscode(vscode):
    def execute(self, *args, **kwargs):
        model = kwargs["model"]

        root_dir = Path("/root")
        config_dir = root_dir / "config"

        inputs_dir = root_dir / "model_dir"
        inputs_dir.mkdir(exist_ok=True)

        dest_name = model.remote_source.split("/")[-1]
        print(f"Downloading file: {dest_name}")
        local_path = model.download()
        with tarfile.open(local_path, "r:gz") as tar:
            print(f"Extracting model to {inputs_dir}")
            tar.extractall(inputs_dir)

        ctx = FlyteContextManager.current_context()
        logger = flytekit.current_context().logging
        ctx.user_space_params.builder().add_attr(
            TASK_FUNCTION_SOURCE_PATH, inspect.getsourcefile(self.task_function)
        ).build()

        # 1. If the decorator is disabled, we don't launch the VSCode server.
        # 2. When user use pyflyte run or python to execute the task, we don't launch the VSCode server.
        #    Only when user use pyflyte run --remote to submit the task to cluster, we launch the VSCode server.
        if not self.enable or ctx.execution_state.is_local_execution():
            return self.task_function(*args, **kwargs)

        if self.run_task_first:
            logger.info("Run user's task first")
            try:
                return self.task_function(*args, **kwargs)
            except Exception as e:
                logger.error(f"Task Error: {e}")
                logger.info("Launching VSCode server")

        # 0. Executes the pre_execute function if provided.
        if self._pre_execute is not None:
            self._pre_execute()
            logger.info("Pre execute function executed successfully!")

        # 1. Downloads the VSCode server from Internet to local.
        download_vscode(self._config)

        # 2. Prepare the interactive debugging Python script and launch.json.
        prepare_interactive_python(self.task_function)  # type: ignore

        # 3. Prepare the task resumption Python script.
        prepare_resume_task_python()

        # 4. Prepare the launch.json
        prepare_launch_json()

        # Move files to config
        files_to_move = [
            ".bashrc",
            ".profile",
            ".wget-hsts",
            "flyteinteractive_interactive_entrypoint.py",
            "flyteinteractive_resume_task.py",
            "script_mode.tar.gz",
            "launch_vscode.py",
        ]

        for file in files_to_move:
            file_path = root_dir / file
            if file_path.exists():
                shutil.move(file_path, config_dir / file)

        # 5. Launches and monitors the VSCode server.
        #    Run the function in the background.
        #    Make the task function's source file directory the default directory.
        task_function_source_dir = os.path.dirname(
            FlyteContextManager.current_context().user_space_params.TASK_FUNCTION_SOURCE_PATH
        )
        child_process = multiprocessing.Process(
            target=execute_command,
            kwargs={
                "cmd": f"code-server --bind-addr 0.0.0.0:{self.port} --disable-workspace-trust --auth none {task_function_source_dir}"
            },
        )
        child_process.start()

        # 6. Register the signal handler for task resumption. This should be after creating the subprocess so that the subprocess won't inherit the signal handler.
        signal.signal(signal.SIGTERM, resume_task_handler)

        return exit_handler(
            child_process=child_process,
            task_function=self.task_function,
            args=args,
            kwargs=kwargs,
            max_idle_seconds=self.max_idle_seconds,
            post_execute=self._post_execute,
        )

    def get_extra_config(self):
        return {self.LINK_TYPE_KEY: VSCODE_TYPE_VALUE, self.PORT_KEY: str(self.port)}


@task(
    container_image=CONTAINER_IMAGE,
    requests=Resources(cpu="3", mem="6Gi"),
)
@custom_vscode
def start_vscode(model: FlyteFile):
    pass


@workflow
def main(model: FlyteFile):
    start_vscode(model=model)
