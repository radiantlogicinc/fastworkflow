import os
from typing import Optional

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer

import fastworkflow


HUGGINGFACE_ENCODER = HuggingFaceEncoder()


class SemanticRouterDefinition:
    def __init__(self, encoder: HuggingFaceEncoder):
        self._encoder = encoder

    def get_route_layer_filepath(self, workflow_folderpath) -> RouteLayer:
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
            workflow_folderpath
        )
        cmddir = command_routing_definition.command_directory
        return os.path.join(
            cmddir.get_commandinfo_folderpath(workflow_folderpath),
            "route_layers.json"
        )

    def get_route_layer(self, workflow_folderpath) -> RouteLayer:
        return RouteLayer.from_json(self.get_route_layer_filepath(workflow_folderpath))

    def train(self, session: fastworkflow.Session) -> RouteLayer:
        workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
            workflow_folderpath
        )
        cmddir = command_routing_definition.command_directory

        routes = []
        utterance_command_tuples = []
        for command_key in cmddir.get_utterance_keys():
            utterance_metadata = cmddir.get_utterance_metadata(command_key)
            utterances_func = utterance_metadata.get_generated_utterances_func(
                    workflow_folderpath
                )
            utterance_list = utterances_func(session)

            command_name = cmddir.get_command_name(command_key)
            routes.append(Route(name=command_name, utterances=utterance_list))

            # dataset for training
            utterance_command_tuples.extend(
                list(zip(utterance_list, [command_name] * len(utterance_list)))
            )

        rl = RouteLayer(encoder=self._encoder, routes=routes)

            # unpack the test data
        X, y = zip(*utterance_command_tuples)
            # evaluate using the default thresholds
        accuracy = rl.evaluate(X=X, y=y)
        print(f"{workflow_folderpath}: Accuracy before training: {accuracy*100:.2f}%")

        threshold_accuracy = 0.1    #TODO: why is training not working?
        if accuracy <= threshold_accuracy:
                # Call the fit method
            rl.fit(X=X, y=y)
                # route_thresholds = rl.get_thresholds()
            accuracy = rl.evaluate(X=X, y=y)
            print(f"{workflow_folderpath}: Accuracy after training: {accuracy*100:.2f}%")
        else:
            print(
                    f"{workflow_folderpath}: Accuracy exceeds {threshold_accuracy*100:.2f}%. No training necessary."
                )

        # save to JSON
        rl.to_json(self.get_route_layer_filepath(workflow_folderpath))
        RouteLayerRegistry._map_workflow_folderpath_to_route_layer[workflow_folderpath] = rl
        return rl

class RouteLayerRegistry:
    @classmethod
    def get_route_layer(cls, workflow_folderpath: str) -> RouteLayer:
        """get the route layer for a given workitem type"""
        if workflow_folderpath not in cls._map_workflow_folderpath_to_route_layer:
            semantic_router_definition = SemanticRouterDefinition(HUGGINGFACE_ENCODER)
            rl = semantic_router_definition.get_route_layer(workflow_folderpath)
            cls._map_workflow_folderpath_to_route_layer[workflow_folderpath] = rl

        return cls._map_workflow_folderpath_to_route_layer[workflow_folderpath]

    @classmethod
    def build_route_layer_from_routelayers(cls, routelayers: list[RouteLayer]) -> RouteLayer:   
        routes = []
        for route_list in routelayers:
            routes.extend(route_list)
        return RouteLayer(encoder=HUGGINGFACE_ENCODER, routes=routes)

    _map_workflow_folderpath_to_route_layer: dict[str, RouteLayer] = {}
