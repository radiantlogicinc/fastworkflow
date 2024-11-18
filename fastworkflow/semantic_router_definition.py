import os

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer

import fastworkflow
from fastworkflow.workflow import Workflow

class SemanticRouterDefinition:
    def __init__(self, encoder: HuggingFaceEncoder, workflow_folderpath: str):
        self._encoder = encoder
        self._workflow_folderpath = workflow_folderpath

    @property
    def workflow_folderpath(self) -> str:
        return self._workflow_folderpath
    
    @property
    def route_layers_folderpath(self) -> str:
        return os.path.join(
            self._workflow_folderpath, "___route_layers"
        )

    def get_route_layer(self, workitem_type: str) -> RouteLayer:
        route_layer_filepath = os.path.join(
            self.route_layers_folderpath, f"{workitem_type}.json"
        )
        return RouteLayer.from_json(route_layer_filepath)

    def train(self, workflow: Workflow):
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(self._workflow_folderpath)
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(self._workflow_folderpath)
        for workitem_type in workflow_definition.types:
            command_names = command_routing_definition.get_command_names(
                workitem_type
            )

            utterance_command_tuples = []

            routes = []
            for command_name in command_names:
                utterance_definition = fastworkflow.UtteranceRegistry.get_definition(self._workflow_folderpath)
                utterances = utterance_definition.get_command_utterances(
                    workitem_type, command_name
                )
                utterances_func = utterances.get_generated_utterances_func(
                    self._workflow_folderpath
                )
                utterance_list = utterances_func(workflow)

                utterance_command_tuples.extend(
                    list(zip(utterance_list, [command_name] * len(utterance_list)))
                )

                routes.append(Route(name=command_name, utterances=utterance_list))

            rl = RouteLayer(encoder=self._encoder, routes=routes)

            # unpack the test data
            X, y = zip(*utterance_command_tuples)
            # evaluate using the default thresholds
            accuracy = rl.evaluate(X=X, y=y)
            print(f"{workitem_type}: Accuracy before training: {accuracy*100:.2f}%")

            threshold_accuracy = 0.1    #TODO: why is training not working?
            if accuracy <= threshold_accuracy:
                # Call the fit method
                rl.fit(X=X, y=y)
                # route_thresholds = rl.get_thresholds()
                accuracy = rl.evaluate(X=X, y=y)
                print(f"{workitem_type}: Accuracy after training: {accuracy*100:.2f}%")
            else:
                print(
                    f"{workitem_type}: Accuracy exceeds {threshold_accuracy*100:.2f}%. No training necessary."
                )

            # save to JSON
            rl.to_json(
                os.path.join(self.route_layers_folderpath, f"{workitem_type}.json")
            )

class RouteLayerRegistry:
    @classmethod
    def build_route_layer_map(cls, workflow_folderpath: str):
        if workflow_folderpath in cls._map_workflow_folderpath_to_route_layer_map:
            return

        route_layers_folderpath = os.path.join(workflow_folderpath, "___route_layers")
        if not os.path.exists(route_layers_folderpath):
            raise ValueError(f"Train the semantic router first. Before running the workflow.")

        encoder = HuggingFaceEncoder()
        semantic_router = SemanticRouterDefinition(encoder, workflow_folderpath)
        map_workitem_type_2_route_layer: dict[str, RouteLayer] = {}

        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
        for workitem_type in workflow_definition.types:
            route_layer = semantic_router.get_route_layer(workitem_type)
            map_workitem_type_2_route_layer[workitem_type] = route_layer
        cls._map_workflow_folderpath_to_route_layer_map[workflow_folderpath] = map_workitem_type_2_route_layer

    @classmethod
    def get_route_layer(cls, workflow_folderpath: str, workitem_type: str) -> RouteLayer:
        """get the route layer for a given workitem type"""
        if workflow_folderpath not in cls._map_workflow_folderpath_to_route_layer_map:
            raise ValueError(f"Route layer map not found for workflow at '{workflow_folderpath}'")
        if workitem_type not in cls._map_workflow_folderpath_to_route_layer_map[workflow_folderpath]:
            raise ValueError(f"Route layer not found for workitem type '{workitem_type}'")
        return cls._map_workflow_folderpath_to_route_layer_map[workflow_folderpath][workitem_type]

    _map_workflow_folderpath_to_route_layer_map: dict[str, dict[str, RouteLayer]] = {}
