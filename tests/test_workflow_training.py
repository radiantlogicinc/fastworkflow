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
        "sample_workflow_path": "./examples/sample_workflow",
        "retail_workflow_path": "./examples/retail_workflow"
    }

def training_artifacts():
    """
    Define training artifacts.
    
    Note: This function is kept for reference but is no longer used directly in the tests.
    The tests now check for artifacts in their actual locations:
    - Context-specific artifacts (in ErrorCorrection and IntentDetection subdirectories)
    - Shared artifacts (directly in the command_info directory)
    """
    # Context-specific artifacts (stored in ErrorCorrection and IntentDetection subdirectories)
    context_specific = [
        "largemodel.pth",
        "tinymodel.pth",
        "ambiguous_threshold.json",
        "threshold.json",
        "label_encoder.pkl",
    ]
    
    # Shared artifacts (stored directly in the command_info directory)
    shared = [
        "command_directory.json",
        "command_routing_definition.json",
    ]
    
    return context_specific + shared

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

    # Purge global cached session databases to avoid stale pointers between tests
    workflow_contexts_path = "./___workflow_contexts"
    if os.path.isdir(workflow_contexts_path):
        shutil.rmtree(workflow_contexts_path)

    if '_workflows' in os.listdir(base_path):
        for child_workflow in os.listdir(os.path.join(base_path, '_workflows')):
            if "__pycache__" in child_workflow:
                continue
            child_workflow_path = os.path.join(base_path, '_workflows', child_workflow)
            if os.path.isdir(child_workflow_path):
                _cleanup_generated_files(child_workflow_path, env_vars)

class TestWorkflowTraining:
    @pytest.mark.skip(reason="Skipping because it takes too long.")
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
        
        # Check for artifacts in the ErrorCorrection and IntentDetection subdirectories
        # These are the actual locations where model files and most artifacts are saved
        error_correction_path = os.path.join(command_info_path, "ErrorCorrection")
        intent_detection_path = os.path.join(command_info_path, "IntentDetection")
        
        # Check that both subdirectories exist
        assert os.path.exists(error_correction_path), f"ErrorCorrection directory not found in {command_info_path}"
        assert os.path.exists(intent_detection_path), f"IntentDetection directory not found in {command_info_path}"
        
        # Check for model files and other artifacts in each subdirectory
        context_artifacts = [
            "largemodel.pth", 
            "tinymodel.pth",
            "ambiguous_threshold.json",
            "threshold.json",
            "label_encoder.pkl"
        ]
        
        for artifact in context_artifacts:
            ec_artifact_path = os.path.join(error_correction_path, artifact)
            id_artifact_path = os.path.join(intent_detection_path, artifact)
            assert os.path.exists(ec_artifact_path), f"{artifact} was not generated in {error_correction_path}."
            assert os.path.exists(id_artifact_path), f"{artifact} was not generated in {intent_detection_path}."
        
        # Check for shared artifacts directly in the command_info directory
        shared_artifacts = [
            "command_directory.json",
            "command_routing_definition.json",
        ]
        for artifact in shared_artifacts:
            artifact_path = os.path.join(command_info_path, artifact)
            assert os.path.exists(artifact_path), f"{artifact} was not generated in {command_info_path}."

    @pytest.mark.skip(reason="Skipping because it takes too long.")
    def test_train_retail_workflow(self, workflow_paths):
        """Tests that training the retail_workflow completes without runtime errors."""
        env_file_path = os.path.join("./env", ".env")
        passwords_file_path = os.path.join("./passwords", ".env")

        env_vars = {
            **dotenv_values(env_file_path),
            **dotenv_values(passwords_file_path)
        }
        # Cleanup existing artifacts for retail_workflow before the test
        _cleanup_generated_files(workflow_paths["retail_workflow_path"], env_vars=env_vars)

        fastworkflow.init(env_vars=env_vars)       
        train_workflow(workflow_paths['retail_workflow_path'])

        # Verify that the expected artifacts were created
        command_info_path = _get_command_info_folderpath(
            workflow_paths["retail_workflow_path"])
            
        # Check for shared artifacts directly in the command_info directory
        # The retail workflow doesn't have ErrorCorrection or IntentDetection contexts
        # as those are part of the core command metadata extraction workflow
        shared_artifacts = [
            "command_directory.json",
            "command_routing_definition.json",
        ]
        for artifact in shared_artifacts:
            artifact_path = os.path.join(command_info_path, artifact)
            assert os.path.exists(artifact_path), f"{artifact} was not generated in {command_info_path}."
