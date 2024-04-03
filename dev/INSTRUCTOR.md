## Instructions

1. Introduce Serverless
2. Setup workspace
    - Use GitHub Codespaces
    - Google Colab
3. Simple workflow
    - Launch workflow from CLI
4. Launch ML Workflow with ImageSpec
    - Code:
        - @task, @workflow, declarative
        - ImageSpec
        - Cache
    - UI:
        - FlyteDeck
        - Show TLM
5. LLM workflow
    - Code:
        - Secrets
        - GPU
        - Show ImageSpec that was used to generate container
    - UI:
        - Use UI to trigger workflow
        - Show TLM for GPUs
        - Show W&B integration
        - Trigger with invalid model to show error
        - Valid models:
            - bert-base-cased
            - distilbert-base-uncased
            - roberta-base
6. VSCode Integration
    - Code:
        - Launch in UI
    - UI:
        - Copy URI from UI for a run
        - Launch from CLI
        - Download model with script
        - Run prompts in the Jupyter
