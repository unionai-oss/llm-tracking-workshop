from textwrap import dedent
from functools import partial
import os
from flytekit.core.context_manager import FlyteContextManager
from flytekit import task, workflow, Resources
from flytekitplugins.flyteinteractive import vscode
from flytekit.types.file import FlyteFile

CONTAINER_IMAGE = "ghcr.io/thomasjpfan/flyte_playground:Mmv063W68q2k3gdZcHImAw"


def construct_file_downloader(task_function):
    task_function_source_path = FlyteContextManager.current_context().user_space_params.TASK_FUNCTION_SOURCE_PATH
    context_working_dir = (
        FlyteContextManager.current_context().execution_state.working_dir
    )
    task_module_name, task_name = task_function.__module__, task_function.__name__
    script = dedent(
        f"""\
    from download import download_model

    download_model("{task_module_name}", "{task_name}", "{context_working_dir}")
    """
    )

    task_function_source_dir = os.path.dirname(task_function_source_path)
    with open(
        os.path.join(task_function_source_dir, "model_downloader.py"), "w"
    ) as file:
        file.write(script)


class custom_vscode(vscode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pre_execute = partial(
            construct_file_downloader, task_function=self.task_function
        )


@task(
    container_image=CONTAINER_IMAGE,
    requests=Resources(cpu="2", mem="12Gi"),
)
@custom_vscode
def start_vscode(model: FlyteFile):
    pass


@workflow
def main(model: FlyteFile):
    start_vscode(model=model)
