import argparse
import os

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
    semantic_router_definition = SemanticRouterDefinition(encoder, args.workflow_folderpath)

    # create a session
    session_id = 1234
    session = Session(session_id, args.workflow_folderpath, args.env_file_path)
    semantic_router_definition.train(session)
