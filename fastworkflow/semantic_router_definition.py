import os
from typing import Optional

import fastworkflow


class SemanticRouterDefinition:

    def get_route_layer_filepath(self, workflow_folderpath) -> str:
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
            workflow_folderpath
        )
        cmddir = command_routing_definition.command_directory
        return os.path.join(
            cmddir.get_commandinfo_folderpath(workflow_folderpath),
            "route_layers.json"
        )

# class RouteLayerRegistry:
#     @classmethod
#     def get_route_layer(cls, workflow_folderpath: str) -> RouteLayer:
#         """get the route layer for a given workitem type"""
#         if workflow_folderpath not in cls._map_workflow_folderpath_to_route_layer:
#             semantic_router_definition = SemanticRouterDefinition(HUGGINGFACE_ENCODER)
#             rl = semantic_router_definition.get_route_layer(workflow_folderpath)
#             cls._map_workflow_folderpath_to_route_layer[workflow_folderpath] = rl

#         return cls._map_workflow_folderpath_to_route_layer[workflow_folderpath]

#     @classmethod
#     def build_route_layer_from_routelayers(cls, routelayers: list[RouteLayer]) -> RouteLayer:   
#         routes = []
#         for route_list in routelayers:
#             routes.extend(route_list)
#         return RouteLayer(encoder=HUGGINGFACE_ENCODER, routes=routes)

#     _map_workflow_folderpath_to_route_layer: dict[str, RouteLayer] = {}
