import argparse
import os
import random
from dotenv import dotenv_values

from colorama import Fore, Style
from semantic_router.encoders import HuggingFaceEncoder

from fastworkflow.semantic_router_definition import SemanticRouterDefinition
from fastworkflow.session import Session

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the semantic router for a workflow"
    )
    parser.add_argument("workflow_folderpath", help="Path to the workflow folder")
    parser.add_argument("env_file_path", help="Path to the environment file")
    args = parser.parse_args()

    if not os.path.isdir(args.workflow_folderpath):
        print(
            f"{Fore.RED}Error: The specified workflow path '{args.workflow_folderpath}' is not a valid directory.{Style.RESET_ALL}"
        )
        exit(1)

    encoder = HuggingFaceEncoder()

    def train_workflow(workflow_path: str, encoder: HuggingFaceEncoder):
        #first, recursively train all child workflows
        workflows_dir = os.path.join(workflow_path, "_workflows")
        if os.path.isdir(workflows_dir):
            for child_workflow in os.listdir(workflows_dir):
                child_workflow_path = os.path.join(workflows_dir, child_workflow)
                if os.path.isdir(child_workflow_path):
                    print(f"{Fore.YELLOW}Training child workflow: {child_workflow_path}{Style.RESET_ALL}")
                    train_workflow(child_workflow_path, encoder)

        # create a session and train the main workflow
        semantic_router_definition = SemanticRouterDefinition(encoder, workflow_path)

        session_id = -random.randint(1, 10000000)
        session = Session(session_id, workflow_path, 
                          env_vars={**dotenv_values(args.env_file_path)}, 
                          for_training_semantic_router=True)
        semantic_router_definition.train(session)
        session.close()

    train_workflow(args.workflow_folderpath, encoder)