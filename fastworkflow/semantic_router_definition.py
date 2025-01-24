import os
from typing import Optional

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer

import fastworkflow


HUGGINGFACE_ENCODER = HuggingFaceEncoder()


def route_layers_folderpath(workflow_folderpath: str) -> str:
    return os.path.join(
        workflow_folderpath, "___route_layers"
    )

class SemanticRouterDefinition:
    def __init__(self, encoder: HuggingFaceEncoder, workflow_folderpath: str):
        self._encoder = encoder
        self._workflow_folderpath = workflow_folderpath

    @property
    def workflow_folderpath(self) -> str:
        return self._workflow_folderpath

    def get_route_layer(self, workitem_path: str) -> RouteLayer:
        route_layer_filepath = os.path.join(
            route_layers_folderpath(self._workflow_folderpath), f"{workitem_path.lstrip('/')}.json"
        )
        return RouteLayer.from_json(route_layer_filepath)

    def train(self, session: fastworkflow.Session):
        workflow = session.workflow_snapshot.workflow
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(self._workflow_folderpath)
        if command_routing_definition := fastworkflow.CommandRoutingRegistry.get_definition(
            self._workflow_folderpath
        ):
            self.build_route_layers(session, command_routing_definition, workflow.path)
            for workitem_path in workflow_definition.paths_2_allowable_child_paths_2_sizemetadata:
                self.build_route_layers(session, command_routing_definition, workitem_path)

    def build_route_layers(self, session, command_routing_definition, workitem_path):
        command_names = command_routing_definition.get_command_names(
                workitem_path
            )

        utterance_command_tuples = []

        routes = []
        for command_name in command_names:
            if command_name == "*":
                continue

            utterance_definition = fastworkflow.UtteranceRegistry.get_definition(self._workflow_folderpath)
            utterances = utterance_definition.get_command_utterances(
                    workitem_path, command_name
                )
            utterances_func = utterances.get_generated_utterances_func(
                    self._workflow_folderpath
                )
            utterance_list = utterances_func(session)

            utterance_command_tuples.extend(
                    list(zip(utterance_list, [command_name] * len(utterance_list)))
                )

            routes.append(Route(name=command_name, utterances=utterance_list))

        rl = RouteLayer(encoder=self._encoder, routes=routes)

            # unpack the test data
        X, y = zip(*utterance_command_tuples)
            # evaluate using the default thresholds
        accuracy = rl.evaluate(X=X, y=y)
        print(f"{workitem_path}: Accuracy before training: {accuracy*100:.2f}%")

        threshold_accuracy = 0.1    #TODO: why is training not working?
        if accuracy <= threshold_accuracy:
                # Call the fit method
            rl.fit(X=X, y=y)
                # route_thresholds = rl.get_thresholds()
            accuracy = rl.evaluate(X=X, y=y)
            print(f"{workitem_path}: Accuracy after training: {accuracy*100:.2f}%")
        else:
            print(
                    f"{workitem_path}: Accuracy exceeds {threshold_accuracy*100:.2f}%. No training necessary."
                )

            # save to JSON
        rl.to_json(
                os.path.join(route_layers_folderpath(self._workflow_folderpath), f"{workitem_path.lstrip('/')}.json")
            )

class RouteLayerRegistry:
    @classmethod
    def get_route_layer(cls, workflow_folderpath: str, workitem_path: str) -> RouteLayer:
        """get the route layer for a given workitem type"""
        if workflow_folderpath in cls._map_workflow_folderpath_to_route_layer_map:
            route_layer_map = cls._map_workflow_folderpath_to_route_layer_map[workflow_folderpath]
        else:
            route_layer_map = cls._build_route_layer_map(workflow_folderpath)
            if not route_layer_map:
                raise ValueError(f"Train the semantic router before running the workflow '{workflow_folderpath}'")

            cls._map_workflow_folderpath_to_route_layer_map[workflow_folderpath] = route_layer_map

        if workitem_path not in route_layer_map:
            raise ValueError(f"Route layer for workitem path {workitem_path} not found.")
        return route_layer_map[workitem_path]

    @classmethod
    def _build_route_layer_map(cls, workflow_folderpath: str) -> Optional[dict[str, RouteLayer]]:
        rl_folderpath = route_layers_folderpath(workflow_folderpath)
        if not os.path.exists(rl_folderpath):
            return None

        semantic_router = SemanticRouterDefinition(HUGGINGFACE_ENCODER, workflow_folderpath)
        map_workitem_path_2_route_layer: dict[str, RouteLayer] = {}

        root_workitem = f"/{os.path.basename(workflow_folderpath.rstrip('/'))}"
        route_layer = semantic_router.get_route_layer(root_workitem)
        map_workitem_path_2_route_layer[root_workitem] = route_layer

        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
        for workitem_path in workflow_definition.paths_2_allowable_child_paths_2_sizemetadata:
            route_layer = semantic_router.get_route_layer(workitem_path)
            map_workitem_path_2_route_layer[workitem_path] = route_layer

        cls._map_workflow_folderpath_to_route_layer_map[workflow_folderpath] = map_workitem_path_2_route_layer
        return map_workitem_path_2_route_layer

    @classmethod
    def build_route_layer_from_routelayers(cls, routelayers: list[RouteLayer]) -> RouteLayer:   
        encoder = HuggingFaceEncoder()
        routes = []
        for route_list in routelayers:
            routes.extend(route_list)
        return RouteLayer(encoder=encoder, routes=routes)

    _map_workflow_folderpath_to_route_layer_map: dict[str, dict[str, RouteLayer]] = {}
