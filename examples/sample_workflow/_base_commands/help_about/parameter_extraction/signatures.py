from typing import Tuple

from pydantic import BaseModel, Field, ConfigDict

import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction


class CommandParameters(BaseModel):
    workitem_path: str = Field(
        default="NOT_FOUND", 
        description="The workitem type",
        json_schema_extra={
            "db_lookup": True
        }
    )



class InputForParamExtraction(BaseModel):
    command: str
    workflow_snapshot: WorkflowSnapshot
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        return cls(command=command, workflow_snapshot=workflow_snapshot)
    
    def db_lookup(self, _:str) -> list[str]: 
        workflow_folderpath = self.workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

        return workflow_definition.paths_2_typemetadata.keys()
        


    def validate_parameters(
        self, cmd_parameters: CommandParameters
    ) -> Tuple[bool, str]:
        """
        Check if the parameters are valid in the current context.
        Parameter is a single field pydantic model.
        Return a tuple with a boolean indicating success or failure.
        And a message with suggested parameter values that are closest matches to the input.
        """
        if cmd_parameters.workitem_path == "NOT_FOUND":
            cmd_parameters.workitem_path = self.workflow_snapshot.active_workitem.path

        return (True, None)
        

# class InputForParamExtraction(BaseModel):
#     command: str
#     current_context: str

#     @classmethod
#     def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
#         cls.__doc__ = (
#             "Given the following list of workitem types: {workitem_types}\n"
#             "Infer the workitem type from the command; and failing that, from the current context\n"
#             "Return the default value if the inferred workitem type is not in the list"
#         )

#         workflow_folderpath = workflow_snapshot.workflow.workflow_folderpath
#         workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
#         workitem_types = ", ".join(workflow_definition.types.keys())
#         cls.__doc__ = cls.__doc__.format(workitem_types=workitem_types)

#         return cls(command=command, current_context=workflow_snapshot.active_workitem.type)
    
#     def db_lookup(self, cmd_parameters:CommandParameters):
#         workflow_folderpath = self.workflow_snapshot.workflow.workflow_folderpath
#         workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
        
#         return workflow_definition.paths_2_typemetadata
        


#     def validate_parameters(
#         self, cmd_parameters: CommandParameters
#     ) -> Tuple[bool, str]:
#         """
#         Check if the parameters are valid in the current context.
#         Parameter is a single field pydantic model.
#         Return a tuple with a boolean indicating success or failure.
#         And a message with suggested parameter values that are closest matches to the input.
#         """
#         if cmd_parameters.workitem_path == "NOT_FOUND":
#             cmd_parameters.workitem_path = self.workflow_snapshot.active_workitem.path
#             return (True, None)
        
#         workflow_folderpath = self.workflow_snapshot.workflow.workflow_folderpath
#         workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
#         if cmd_parameters.workitem_path in workflow_definition.paths_2_typemetadata:
#             return (True, None)

        # workitem_types = "\n".join(workflow_definition.paths_2_typemetadata.keys())
        # return (
        #     False,
        #     (
        #         f"The workitem type {cmd_parameters.workitem_path} is not in the list of valid workitem types:\n"
        #         f"{workitem_types}\n"
        #         "Please choose a valid workitem type from the list"
        #     ),
        # )

    # class Config:
    #     arbitrary_types_allowed = True



