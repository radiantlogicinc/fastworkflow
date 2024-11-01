import os
import json
from typing import Optional
from enum import Enum

from pydantic import BaseModel, field_validator
from pydantic import model_validator

class NodeType(str, Enum):
    Workitem = "Workitem"
    Workflow = "Workflow"

class TypeMetadata(BaseModel):
    node_type: NodeType

    @field_validator('node_type', mode='before')
    def parse_node_type(cls, node_type: NodeType):
        if not node_type:
            raise ValueError("node_type cannot be empty")
        return node_type

class SizeMetaData(BaseModel):
    min: int
    max: Optional[int]

    @field_validator('min', mode='before')
    def parse_min(cls, min: int):
        if min is None:
            raise ValueError("Minimum value cannot be empty")

        if min < 0:
            raise ValueError("Minimum value must be greater than or equal to 0")

        return min

    @field_validator('max', mode='before')
    def parse_max(cls, max: Optional[int]):
        if max is not None:
            if max < 1:
                raise ValueError("Maximum value must be greater than or equal to 1")
        return max

    @model_validator(mode='after')
    def check_size_metadata(cls, size_meta: 'SizeMetaData'):
        if size_meta.max is not None:
            if size_meta.min > size_meta.max:
                raise ValueError("Maximum value must be greater than or equal to the minimum value")
        return size_meta

class WorkflowDefinition(BaseModel):
    types: dict[str, TypeMetadata]
    allowable_child_types: dict[str, dict[str, SizeMetaData]]  

    @field_validator('types', mode='before')
    def parse_type_metadata(cls, types: dict[str, TypeMetadata]):
        for key, value in types.items():
            if isinstance(value, dict):
                types[key] = TypeMetadata(**value)
            elif not isinstance(types[key], TypeMetadata):
                raise ValueError(f"Invalid value for type metadata '{key}'")
        return types

    @field_validator('allowable_child_types', mode='before')
    def parse_size_metadata(cls, allowable_child_types: dict[str, dict[str, SizeMetaData]]):
        for _, children in allowable_child_types.items():
            for child_type, size_meta in children.items():
                if isinstance(size_meta, dict):
                    children[child_type] = SizeMetaData(**size_meta)
                elif not isinstance(children[child_type], SizeMetaData):
                    raise ValueError(f"Invalid value for child size metadata '{child_type}'")
        return allowable_child_types

    @model_validator(mode='after')
    def check_workflow_definition(cls, wfd: 'WorkflowDefinition'):
        # check that all types have a valid non-empty key
        for key in wfd.types.keys():
            if not key:
                raise ValueError("Workflow/workitem type cannot be an empty string")
            
        for parent_type, children in wfd.allowable_child_types.items():
            if parent_type not in wfd.types:
                raise ValueError(f"Parent type '{parent_type}' is not defined in types")
            for child_type, size_metadata in children.items():
                if child_type not in wfd.types:
                    raise ValueError(f"Child type '{child_type}' is not defined in types")
        return wfd

    @classmethod
    def create(cls, workflow_folderpath: str) -> 'WorkflowDefinition':
        types = {}
        allowable_child_types = {}

        cls._populate_workflow_definition(workflow_folderpath, types, allowable_child_types)

        workflow_definition = WorkflowDefinition(
            types=types,
            allowable_child_types=allowable_child_types
        )

        return workflow_definition

    @classmethod
    def _populate_workflow_definition(
        cls,
        workflow_folderpath: str,
        types: dict[str, TypeMetadata],
        allowable_child_types: dict[str, dict[str, SizeMetaData]]
    ):
        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        workitem_type = os.path.basename(workflow_folderpath.rstrip('/'))
        # if workitem_type.startswith("_"):
        #     raise ValueError(f"{workitem_type} starts with an '_'. Names starting with an _ are reserved")

        types[workitem_type] = TypeMetadata(node_type = NodeType.Workitem)
        if workitem_type not in allowable_child_types:
            allowable_child_types[workitem_type] = {}

        # Recursively process subfolders
        for subfolder in os.listdir(workflow_folderpath):
            subfolder_path = os.path.join(workflow_folderpath, subfolder)
            if os.path.isdir(subfolder_path) and not subfolder.startswith("_"):
                types[workitem_type] = TypeMetadata(node_type = NodeType.Workflow)
                child_workitem_type = os.path.basename(subfolder_path.rstrip('/'))
                allowable_child_types[workitem_type][child_workitem_type] = SizeMetaData(min=0, max=None)
                cls._populate_workflow_definition(subfolder_path, types, allowable_child_types)

        # Read the child cardinality if it exists
        child_cardinality_file = os.path.join(workflow_folderpath, "child_cardinality.json")
        if os.path.exists(child_cardinality_file):
            with open(child_cardinality_file, 'r') as f:
                child_cardinality = json.load(f)
                for child_type, size_meta in child_cardinality.items():
                    if child_type not in allowable_child_types[workitem_type]:
                        raise ValueError("cardinality file contains a child of type {child_type} that does not exist for {workitem_type}")
                    allowable_child_types[workitem_type][child_type] = SizeMetaData(**size_meta)

    def write(self, filename: str):
        with open (filename, 'w') as f:
            f.write(self.model_dump_json(indent=4))

    class Config:
        arbitrary_types_allowed = True
