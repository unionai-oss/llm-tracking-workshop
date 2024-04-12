from subprocess import run
from time import sleep
import re
from flytekit.models.core import execution as core_execution_models


def get_recent_execution_id(workflow_name, remote):
    recent_executions = remote.recent_executions()
    executions = [
        e for e in recent_executions if e.spec.launch_plan.name == workflow_name
    ]

    return executions[0].id.name


def launch_vscode(llm_uri, remote):
    command = [
        "unionai",
        "run",
        "--remote",
        "workflows/launch_vscode.py",
        "main",
        "--model",
        llm_uri,
    ]
    result = run(command, capture_output=True, text=True)
    match = re.search(r"executions/(\w+) to see execution", result.stdout)

    if not match:
        raise RuntimeError("Please run launch_vscode again")

    execution_id = match.group(1)
    print("VSCode workflow launched! Waiting for VSCode server to start...")

    execution = remote.fetch_execution(name=execution_id)
    execution = remote.sync_execution(execution, sync_nodes=True)
    vscode_logs = []
    count = 0

    while not vscode_logs:
        count += 1
        if count % 10 == 0:
            print("Still waiting...")

        execution = remote.sync_execution(execution, sync_nodes=True)
        if execution.is_done and execution.error is not None:
            print("There was an error launching VScode. Please try again.")

        if execution.closure.phase != core_execution_models.TaskExecutionPhase.RUNNING:
            sleep(1)
            continue

        if not execution.node_executions or "n0" not in execution.node_executions:
            sleep(1)
            continue

        node_exectuion = execution.node_executions["n0"]
        task_execution = node_exectuion.task_executions[0]
        logs = task_execution.closure.logs

        vscode_logs = [l for l in logs if l.name.startswith("VSCode")]
        sleep(1)

    for _ in range(3):
        print("Still waiting...")
        sleep(10)

    uri = vscode_logs[0].uri
    endpoint = remote.config.platform.endpoint
    print("✅ Your VSCode instance launched at:")
    print(f"https://{endpoint}{uri}")

    return execution.id.name
