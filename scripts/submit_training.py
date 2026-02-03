# scripts/submit_training.py
"""
Submit training job to Azure ML from GitHub Actions
"""
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

def main():
    # Get credentials from environment variables
    subscription_id = os.environ['SUBSCRIPTION_ID']
    resource_group = os.environ['RESOURCE_GROUP']
    workspace_name = os.environ['WORKSPACE_NAME']
    
    # Connect to workspace
    print("Connecting to Azure ML workspace...")
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name
    )
    print(f"Connected to: {ws.name}")
    
    # Get or create compute cluster
    compute_name = 'cpu-cluster'
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print(f"Found existing cluster: {compute_name}")
    except ComputeTargetException:
        print(f"Creating new cluster: {compute_name}")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='STANDARD_DS2_V2',
            min_nodes=0,
            max_nodes=2
        )
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=True)
    
    # Create environment
    env = Environment.from_pip_requirements(
        name='training-env',
        file_path='requirements.txt'
    )
    
    # Configure training run
    src = ScriptRunConfig(
        source_directory='src',
        script='train.py',
        compute_target=compute_target,
        environment=env
    )
    
    # Submit experiment
    experiment = Experiment(ws, 'github-actions-training')
    run = experiment.submit(src)
    print(f"Submitted run: {run.id}")
    print(f"Run URL: {run.get_portal_url()}")
    
    # Wait for completion
    run.wait_for_completion(show_output=True)
    print(f"Run status: {run.get_status()}")

if __name__ == '__main__':
    main()