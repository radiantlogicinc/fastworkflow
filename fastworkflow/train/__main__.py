import argparse
import os
from dotenv import dotenv_values

from colorama import Fore, Style

import fastworkflow
from fastworkflow.model_pipeline_training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the intent detection pipeline for a workflow"
    )
    parser.add_argument("workflow_folderpath", help="Path to the workflow folder")
    parser.add_argument("env_file_path", help="Path to the environment file")
    args = parser.parse_args()

    if not os.path.isdir(args.workflow_folderpath):
        print(
            f"{Fore.RED}Error: The specified workflow path '{args.workflow_folderpath}' is not a valid directory.{Style.RESET_ALL}"
        )
        exit(1)

    fastworkflow.init(env_vars={**dotenv_values(args.env_file_path)})

    def train_workflow(workflow_path: str):
        fastworkflow.WorkflowRegistry.create_definition(workflow_path)
        fastworkflow.CommandRoutingRegistry.create_definition(workflow_path)
        fastworkflow.UtteranceRegistry.create_definition(workflow_path)

        #first, recursively train all child workflows
        workflows_dir = os.path.join(workflow_path, "_workflows")
        if os.path.isdir(workflows_dir):
            for child_workflow in os.listdir(workflows_dir):
                if "__pycache__" in child_workflow:
                    continue
                child_workflow_path = os.path.join(workflows_dir, child_workflow)
                if os.path.isdir(child_workflow_path):
                    print(f"{Fore.YELLOW}Training child workflow: {child_workflow_path}{Style.RESET_ALL}")
                    train_workflow(child_workflow_path)

        if workflow_path.startswith("./fastworkflow") and "_workflows" not in workflow_path:
            return

        # create a session and train the main workflow
        session = fastworkflow.Session.create(
            workflow_path, 
            session_id_str=f"train_{workflow_path}", 
            for_training_semantic_router=True
        )

        train(session)

        session.close()

    train_workflow(args.workflow_folderpath)