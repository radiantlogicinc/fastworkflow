import argparse
import os

from colorama import Fore, Style
from semantic_router.encoders import HuggingFaceEncoder

from fastworkflow.session import Session
from fastworkflow.semantic_router_definition import SemanticRouterDefinition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the semantic router for a workflow")
    parser.add_argument("workflow_folderpath", help="Path to the workflow folder")
    args = parser.parse_args()

    if not os.path.isdir(args.workflow_folderpath):
        print(f"{Fore.RED}Error: The specified workflow path '{args.workflow_path}' is not a valid directory.{Style.RESET_ALL}")
        exit(1)

    # create a session
    session_id = 1234

    session = Session(session_id, args.workflow_folderpath)

    encoder = HuggingFaceEncoder()
    semantic_router_definition = SemanticRouterDefinition(session, encoder)
    semantic_router_definition.train()
