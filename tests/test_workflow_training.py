import os
import shutil
import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.train.__main__ import train_workflow


@pytest.fixture
def workflow_paths():
    """Define paths for fastworkflow and sample_workflow."""
    return {
        "fastworkflow_package_path": fastworkflow.get_fastworkflow_package_path(),
        "sample_workflow_path": "./examples/sample_workflow"
    }

def training_artifacts():
    """Define training artifacts."""
    return [
        "largemodel.pth",
        "tinymodel.pth",
        "ambiguous_threshold.json",
        "command_directory.json",
        "command_routing_definition.json",
        "label_encoder.pkl",
        "threshold.json",
    ]

def _get_command_info_folderpath(base_path):
    """Get the ___command_info folder path"""
    return os.path.join(base_path, "___command_info")

def _cleanup_generated_files(base_path, env_vars=None):
    """Removes all generated artifact files in the ___command_info subfolder."""
    command_info_path = _get_command_info_folderpath(base_path)

    if os.path.isdir(command_info_path):
        shutil.rmtree(command_info_path)
    
    speeddict_folder_name = env_vars["SPEEDDICT_FOLDERNAME"]
    if os.path.isdir(os.path.join(base_path, speeddict_folder_name)):
        shutil.rmtree(os.path.join(base_path, speeddict_folder_name))

    if '_workflows' in os.listdir(base_path):
        for child_workflow in os.listdir(os.path.join(base_path, '_workflows')):
            if "__pycache__" in child_workflow:
                continue
            child_workflow_path = os.path.join(base_path, '_workflows', child_workflow)
            if os.path.isdir(child_workflow_path):
                _cleanup_generated_files(child_workflow_path, env_vars)

class TestWorkflowTraining:
    def test_train_fastworkflow(self, workflow_paths):
        """Tests that training fastworkflow base generates the proper set of artifacts."""
        
        env_file_path = os.path.join("./env", ".env")
        passwords_file_path = os.path.join("./passwords", ".env")

        env_vars = {
            **dotenv_values(env_file_path),
            **dotenv_values(passwords_file_path)
        }
        
        # Cleanup any existing artifacts before the test
        _cleanup_generated_files(
            workflow_paths["fastworkflow_package_path"], 
            env_vars=env_vars
        )

        fastworkflow.init(env_vars=env_vars)
        train_workflow(workflow_paths["fastworkflow_package_path"])

        # Verify that the expected artifacts were created
        command_info_path = _get_command_info_folderpath(
            os.path.join(
                workflow_paths["fastworkflow_package_path"], 
                '_workflows',
                'command_metadata_extraction'
            )
        )
# sourcery skip: no-loop-in-tests
        for artifact in training_artifacts():
            artifact_path = os.path.join(command_info_path, artifact)
            assert os.path.exists(artifact_path), f"{artifact} was not generated in {command_info_path}."

    def test_train_sample_workflow(self, workflow_paths):
        """Tests that training the sample_workflow completes without runtime errors."""
        env_file_path = os.path.join("./env", ".env")
        passwords_file_path = os.path.join("./passwords", ".env")

        env_vars = {
            **dotenv_values(env_file_path),
            **dotenv_values(passwords_file_path)
        }
        # Cleanup existing artifacts for sample_workflow before the test
        _cleanup_generated_files(workflow_paths["sample_workflow_path"], env_vars=env_vars)

        fastworkflow.init(env_vars=env_vars)       
        train_workflow(workflow_paths['sample_workflow_path'])

        # Verify that the expected artifacts were created
        command_info_path = _get_command_info_folderpath(
            workflow_paths["sample_workflow_path"])
# sourcery skip: no-loop-in-tests
        for artifact in training_artifacts():
            artifact_path = os.path.join(command_info_path, artifact)
            assert os.path.exists(artifact_path), f"{artifact} was not generated in {command_info_path}."
