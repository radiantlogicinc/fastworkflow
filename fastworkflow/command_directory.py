import ast
import contextlib
import functools
import hashlib
import os
import re
from typing import Optional
from pathlib import Path
from functools import lru_cache

from pydantic import BaseModel, field_validator, model_validator, ConfigDict, Field, ValidationInfo

from fastworkflow.utils import python_utils
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_inheritance_model import WorkflowInheritanceModel


class CommandMetadata(BaseModel):
    """Command metadata"""
    # parameter extraction
    workflow_folderpath: Optional[str] = None
    parameter_extraction_signature_module_path: Optional[str] = None
    input_for_param_extraction_class: Optional[str] = None
    command_parameters_class: Optional[str] = None
    # response generation
    response_generation_module_path: str
    response_generation_class_name: str
    is_planning: bool = False

    @model_validator(mode="after")
    def validate_parameter_extraction_fields(self) -> "CommandMetadata":
        rg_fields = [
            self.response_generation_module_path,
            self.response_generation_class_name,
        ]

        if not (all(rg_fields)):
            raise ValueError("Response generation modules must be populated.")

        return self


#: ``UtteranceMetadata.generated_utterances_func_name`` of a command whose ``Signature``
#: defines no ``generate_utterances``: the framework's default (``default_generate_utterances``).
DEFAULT_UTTERANCES_FUNC = "<default>"


def default_generate_utterances(signature, workflow, command_name: str) -> list[str]:
    """The utterances of a command whose ``Signature`` defines no ``generate_utterances``:
    exactly what the standard command-file body returns (``build/command_file_template``):
    the command's name in words, then ``generate_diverse_utterances`` of its
    ``plain_utterances``. A ``Signature`` that defines (or inherits) ``generate_utterances``
    keeps its own; this is never used for it."""
    from fastworkflow.train import generate_synthetic

    return [command_name.split("/")[-1].lower().replace("_", " ")] + \
        generate_synthetic.generate_diverse_utterances(
            getattr(signature, "plain_utterances", []), command_name)


class UtteranceMetadata(BaseModel):
    workflow_folderpath: Optional[str] = None
    plain_utterances: list[str]
    template_utterances: list[str]
    generated_utterances_module_filepath: str
    generated_utterances_func_name: str

    def get_generated_utterances_func(self, workflow_folderpath: str) -> list[str]:      
        module = python_utils.get_module(
            self.generated_utterances_module_filepath, 
            self.workflow_folderpath or workflow_folderpath
        )

        # Get the function from the module and execute it
        if not module or not self.generated_utterances_func_name:
            return None

        if self.generated_utterances_func_name == DEFAULT_UTTERANCES_FUNC:
            signature = getattr(module, "Signature", None)
            return functools.partial(default_generate_utterances, signature) if signature else None

        if '.' in self.generated_utterances_func_name:
            parts = self.generated_utterances_func_name.split('.')
            func = module
            for part in parts:
                func = getattr(func, part)
            return func

        return getattr(module, self.generated_utterances_func_name, None)

    @field_validator("plain_utterances", mode="before")
    def parse_plain_utterances(cls, plain_utterances: list[str]):
        for s in plain_utterances:
            if not s:
                raise ValueError("Plain utterance string cannot be empty")
        return plain_utterances

    @field_validator("template_utterances", mode="before")
    def parse_template_utterances(cls, template_utterances: list[str]):
        for s in template_utterances:
            if not s:
                raise ValueError("Plain utterance string cannot be empty")
        return template_utterances

    @field_validator("generated_utterances_module_filepath", mode="before")
    def parse_generated_utterances_module_path(
        cls, generated_utterances_module_filepath: str
    ):
        if not generated_utterances_module_filepath:
            raise ValueError("Generated utterances module path cannot be empty")
        return generated_utterances_module_filepath

    @field_validator("generated_utterances_func_name", mode="before")
    def parse_generatd_utterances_func_name(cls, generated_utterances_func_name: str):
        if not generated_utterances_func_name:
            raise ValueError("Generated utterances function name cannot be empty")
        return generated_utterances_func_name


class ContextMetadata(BaseModel):
    """Metadata describing a context callback implementation module."""

    workflow_folderpath: Optional[str] = None
    context_module_path: str
    context_class: str

    @field_validator("context_module_path", mode="before")
    def validate_context_module_path(cls, v):
        if not v:
            raise ValueError("context_module_path cannot be empty")
        return v

    @field_validator("context_class", mode="before")
    def validate_context_class(cls, v):
        if not v:
            raise ValueError("context_class cannot be empty")
        return v


class CommandDirectory(BaseModel):
    """
    Centralized directory for all command implementations.
    Each command is uniquely identified by its name.
    """
    workflow_folderpath: str

    map_command_2_metadata: dict[str, CommandMetadata] = Field(default_factory=dict)
    map_command_2_utterance_metadata: dict[str, UtteranceMetadata] = Field(default_factory=dict)
    core_command_names: list[str] = Field(default_factory=list)
    # Mapping of context name to its metadata containing callback class module information
    map_context_2_metadata: dict[str, "ContextMetadata"] = Field(default_factory=dict)
    # Fingerprint of the command sources this snapshot was built from. Used to
    # invalidate the persisted cache when commands are added/removed/renamed/edited.
    source_fingerprint: Optional[str] = None

    def register_command_metadata(self, command_name: str, metadata: CommandMetadata):
        if not (command_name and metadata):
            raise ValueError("command_key and metadata are required")
        self.map_command_2_metadata[command_name] = metadata

    def get_commands(self) -> list[str]:
        """Retrieve all command keys registered in the command directory."""
        return list(self.map_command_2_metadata.keys())

    def get_execution_commands(self) -> list[str]:
        """Command names that are not planning commands (routing, training, catalogs)."""
        return [
            name
            for name, metadata in self.map_command_2_metadata.items()
            if not metadata.is_planning
        ]

    def get_command_metadata(self, command_name: str) -> CommandMetadata:
        if command_name not in self.map_command_2_metadata:
            raise KeyError(f"Command key '{command_name}' not found.")
        return self.map_command_2_metadata[command_name]

    def register_utterance_metadata(self, command_name: str, metadata: UtteranceMetadata):
        if not (command_name and metadata):
            raise ValueError("command_key and metadata are required")
        self.map_command_2_utterance_metadata[command_name] = metadata

    def get_utterance_keys(self) -> list[str]:
        """Retrieve all utterance keys registered in the command directory."""
        return list(self.map_command_2_utterance_metadata.keys())

    def get_utterance_metadata(self, command_key: str) -> Optional[UtteranceMetadata]:
        if meta := self.map_command_2_utterance_metadata.get(command_key):
            return meta

        # -------- lazy hydration --------
        if command_key in self.map_command_2_metadata:
            self.ensure_command_hydrated(command_key)
            if meta := self.map_command_2_utterance_metadata.get(command_key):
                return meta
        
        return None

    @classmethod
    def get_commandinfo_folderpath(cls, workflow_folderpath: str) -> str:
        command_info_folderpath = Path(workflow_folderpath) / "___command_info"
        command_info_folderpath.mkdir(parents=True, exist_ok=True)
        return str(command_info_folderpath)

    def save(self):
        commandroutinginfo_folderpath = CommandDirectory.get_commandinfo_folderpath(self.workflow_folderpath)
        # Stamp the current source fingerprint so the persisted snapshot can be
        # validated (and invalidated) against the live _commands tree on reload.
        self.source_fingerprint = compute_commands_source_fingerprint(self.workflow_folderpath)
        with open(Path(commandroutinginfo_folderpath) / "command_directory.json", "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(
        cls,
        workflow_folderpath: str,
        *,
        hydrate_utterances: bool = True,
        best_effort_inheritance: bool = False,
    ) -> "CommandDirectory":
        """
        Loads the command directory by scanning _commands folders from base workflows 
        and the current workflow, implementing workflow inheritance.
        
        Search order (last occurrence wins):
        1. Each path in workflow_inheritance_model.json["base"] (in order) -> their _commands folders
        2. <workflow_root>/_commands (current workflow - highest precedence)
        3. Built-in core workflow fastworkflow/_workflows/command_metadata_extraction
        
        Commands in context subdirectories are registered with qualified names 
        in the format 'ContextName/command_name'.
        """
        command_directory = cls(workflow_folderpath=workflow_folderpath)
        
        # Step 1: Load workflow inheritance model
        try:
            inheritance_model = WorkflowInheritanceModel.load(workflow_folderpath)
            base_paths = inheritance_model.resolve_base_paths(workflow_folderpath)
        except Exception as e:
            if best_effort_inheritance:
                base_paths = []
            else:
                raise RuntimeError(f"Error processing workflow inheritance: {e}") from e

        # A retired _skills/ folder (here or in a base) is a load error: its planning commands are not read
        for path in (*base_paths, workflow_folderpath):
            check_no_planning_commands_folder(path)
        
        # Step 2: Process base workflows first (in order of precedence)
        for base_path in base_paths:
            cls._load_commands_from_path(command_directory, base_path)
        
        # Step 3: Process current workflow commands (highest precedence for user commands)
        cls._load_commands_from_path(command_directory, workflow_folderpath)

        # Step 4: Populate utterance metadata when available (optional; defers imports)
        if hydrate_utterances:
            for command_key in command_directory.map_command_2_metadata:
                if utterance_metadata := command_directory.get_utterance_metadata(command_key):
                    command_directory.register_utterance_metadata(command_key, utterance_metadata)

        # Step 5: Register core commands (processed last, lowest precedence)
        cls._register_core_commands(command_directory)

        return command_directory

    @classmethod
    def scan(cls, workflow_folderpath: str) -> "CommandDirectory":
        """Load command metadata without hydrating utterances (no imports, no ``___command_info``)."""
        return cls.load(
            workflow_folderpath,
            hydrate_utterances=False,
            best_effort_inheritance=True,
        )
    
    @classmethod
    def _load_commands_from_path(cls, command_directory: "CommandDirectory", path: str):
        """
        Load commands from a specific workflow path.
        
        This method handles loading commands from both base workflows and the current workflow.
        Commands are registered with the command directory, with later calls overriding earlier ones
        for the same command name (implementing precedence).
        
        Args:
            command_directory: The CommandDirectory instance to register commands to
            path: Path to the workflow folder containing _commands/
        """
        commands_root_folder = Path(path) / "_commands"
        
        if not commands_root_folder.is_dir():
            # Skip silently if no _commands directory exists
            # This allows for base workflows that may not have commands
            return
            
        # Process files directly under _commands/ (global commands)
        for item_path in commands_root_folder.glob("*.py"):
            if item_path.is_file() and not item_path.name.startswith("_") and item_path.name != "__init__.py":
                command_name = item_path.stem
                cls._register_command(command_directory, command_name, item_path, path)
        
        # Process subdirectories (contexts) and their commands
        for context_dir in commands_root_folder.iterdir():
            if context_dir.is_dir() and not context_dir.name.startswith("_"):
                context_name = context_dir.name
                
                # Check if this context directory contains any command files
                # (execution or planning — a planning-only context is not empty)
                has_commands = False
                
                for command_file in context_dir.glob("*.py"):
                    if not command_file.name.startswith("_") and command_file.name != "__init__.py":
                        has_commands = True
                        command_name = command_file.stem
                        # Use qualified name format: "ContextName/command_name"
                        qualified_command_name = f"{context_name}/{command_name}"
                        cls._register_command(command_directory, qualified_command_name, command_file, path)
                
                # For the current workflow (not base workflows), enforce the no-empty-contexts rule
                if not has_commands and path == command_directory.workflow_folderpath:
                    # This is an empty context folder in the main workflow - flag as error
                    raise ValueError(
                        f"Context folder '{context_name}' exists but contains no command files. "
                        f"Empty context folders are not allowed in the main workflow."
                    )
                
                # After processing commands, attempt to register context class implementation if present
                if has_commands:  # Only register context classes if there are actual commands
                    cls._register_context_class(command_directory, context_name, context_dir, path)
    
    @staticmethod
    def check_planning_command_path(name: str, path: str) -> None:
        """Validate a planning command's file name before load: its stem is a Python identifier.

        Raises :class:`fastworkflow.planning_commands.PlanningCommandFormatError` when it is not.
        """
        # planning_commands imports command_directory at load (via command_context_model); import here to break the cycle
        from fastworkflow.planning_commands import PlanningCommandFormatError

        _context, _, stem = name.rpartition("/")
        if not stem.isidentifier():
            raise PlanningCommandFormatError(
                f"{path}: a planning command's name is its file name, a Python identifier "
                f"(snake_case): rename it to {re.sub(r'[^0-9A-Za-z_]', '_', stem)}.py"
            )

    @classmethod
    def _register_command(
        cls,
        command_directory: "CommandDirectory",
        command_name: str,
        command_file_path: Path,
        workflow_folderpath: str,
        *,
        response_generation_class_name: str | None = None,
        is_planning: bool | None = None,
    ):
        """Helper method to register a command and its metadata"""
        command_filepath_str = str(command_file_path)

        # ------------------------------------------------------------
        # Lazy-loading variant: we avoid importing the command module
        # here.  Only the bare minimum metadata needed for routing is
        # stored up-front; Signature/utterance data is hydrated the first
        # time it is requested (see `ensure_command_hydrated`).
        # ------------------------------------------------------------

        if response_generation_class_name is None or is_planning is None:
            rg_name, detected_planning = _command_kind_from_ast(command_file_path)
            if response_generation_class_name is None:
                response_generation_class_name = rg_name
            if is_planning is None:
                is_planning = detected_planning

        metadata = CommandMetadata(
            workflow_folderpath=workflow_folderpath,
            # Defer Signature import – fill these lazily
            parameter_extraction_signature_module_path=None,
            input_for_param_extraction_class=None,
            command_parameters_class=None,
            # Response generation path is still required immediately
            response_generation_module_path=command_filepath_str,
            response_generation_class_name=response_generation_class_name,
            is_planning=is_planning,
        )
        command_directory.register_command_metadata(command_name, metadata)

    @staticmethod
    def _populate_utterance_metadata_for_command(
        command_directory, command_name, command_metadata, workflow_folderpath, module=None
    ):
        """Helper to extract utterance info from a command's Signature."""
        if module is None:
            module = python_utils.get_module(
                command_metadata.response_generation_module_path,
                command_metadata.workflow_folderpath or workflow_folderpath,
            )
        if not module:
            return

        Signature = getattr(module, "Signature", None)
        if not Signature:
            return

        # A Signature that defines (or inherits) generate_utterances keeps it; one that does
        # not gets the framework default, the standard body (default_generate_utterances).
        plain_utterances = getattr(Signature, "plain_utterances", [])
        template_utterances = getattr(Signature, "template_utterances", [])

        # The command file stays the source path: training fingerprints hash it.
        generated_utterances_module_filepath = command_metadata.response_generation_module_path
        generated_utterances_func_name = (
            "Signature.generate_utterances" if hasattr(Signature, "generate_utterances")
            else DEFAULT_UTTERANCES_FUNC)

        utterance_metadata = UtteranceMetadata(
            workflow_folderpath=command_metadata.workflow_folderpath or workflow_folderpath,
            plain_utterances=plain_utterances,
            template_utterances=template_utterances,
            generated_utterances_module_filepath=generated_utterances_module_filepath,
            generated_utterances_func_name=generated_utterances_func_name,
        )
        command_directory.register_utterance_metadata(command_name, utterance_metadata)

    @field_validator("map_command_2_metadata", mode="before")
    def validate_map_command_2_metadata(cls, v, info: ValidationInfo):
        # This validator is primarily for when CommandDirectory is created from a dict (e.g. JSON)
        # In our `load` method, we construct CommandMetadata objects directly.
        if not isinstance(v, dict):
            raise ValueError("map_command_2_metadata must be a dictionary.")
        
        workflow_folderpath = info.data.get('workflow_folderpath')
        
        for command_key, metadata_val in v.items():
            if not command_key:
                raise ValueError("Command key cannot be an empty string.")
            if isinstance(metadata_val, dict):
                # Ensure workflow_folderpath is present if CommandMetadata is being created from a dict
                if 'workflow_folderpath' not in metadata_val and workflow_folderpath:
                     metadata_val['workflow_folderpath'] = workflow_folderpath
                v[command_key] = CommandMetadata(**metadata_val)
            elif not isinstance(metadata_val, CommandMetadata):
                raise ValueError(f"Invalid metadata for command key '{command_key}'")
        return v

    @field_validator("map_command_2_utterance_metadata", mode="before")
    def parse_map_command_2_utterances(
        cls, v: dict[str, UtteranceMetadata], info: ValidationInfo
    ):
        # Similar to above, mainly for dict -> model conversion
        if not isinstance(v, dict):
            raise ValueError("map_command_2_utterance_metadata must be a dictionary.")
            
        workflow_folderpath = info.data.get('workflow_folderpath')
        for key, value in v.items():
            if isinstance(value, dict):
                if 'workflow_folderpath' not in value and workflow_folderpath:
                    value['workflow_folderpath'] = workflow_folderpath
                v[key] = UtteranceMetadata(**value)
            elif not isinstance(v[key], UtteranceMetadata): # Check v[key]
                raise ValueError(f"Invalid value for type metadata '{key}'")
        return v

    @staticmethod
    def _register_core_commands(command_directory: "CommandDirectory") -> None:
        """Dynamically discover and register core commands from the command_metadata_extraction workflow.

        Rules:
        1.  Core commands come from *all* python modules inside the internal
            ``command_metadata_extraction/_commands`` tree (recursively).
        """
        # Clear any previously populated list first
        command_directory.core_command_names.clear()

        import fastworkflow
        internal_wf_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
        internal_cmd_root = os.path.join(internal_wf_path, "_commands")
        
        # Get the discovered commands using the cached function
        discovered_commands = CommandDirectory._discover_core_commands(internal_wf_path)
        
        # Store sorted list
        command_directory.core_command_names.extend(sorted(discovered_commands))

        # ------------------------------------------------------------
        # Register each core command's metadata & utterances (unless
        # already registered in this directory build).
        # ------------------------------------------------------------
        for qualified_cmd in command_directory.core_command_names:
            # Determine the actual file path inside the internal workflow
            # irrespective of current workflow.
            if "/" in qualified_cmd:
                ctx, cmd = qualified_cmd.split("/", 1)
                module_rel_path = os.path.join(ctx, f"{cmd}.py")
            else:
                module_rel_path = f"{qualified_cmd}.py"

            module_path = os.path.join(internal_cmd_root, module_rel_path)

            # Try to inspect module for Signature presence
            module = python_utils.get_module(module_path, internal_wf_path)
            sig_exists = hasattr(module, "Signature") if module else False
            input_exists = bool(sig_exists and hasattr(module.Signature, "Input"))

            metadata = CommandMetadata(
                workflow_folderpath=internal_wf_path,
                parameter_extraction_signature_module_path=(module_path if sig_exists else None),
                input_for_param_extraction_class=("Signature" if sig_exists else None),
                command_parameters_class=("Signature.Input" if input_exists else None),
                response_generation_module_path=module_path,
                response_generation_class_name="ResponseGenerator",
            )
            command_directory.register_command_metadata(qualified_cmd, metadata)

        # Core contexts' callback classes (e.g. `occupiable = False` on IntentDetection)
        # are registered with their commands; the scan above does not reach them.
        for context_name in sorted({name.split("/")[0] for name in discovered_commands if "/" in name}):
            CommandDirectory._register_context_class(
                command_directory, context_name,
                Path(internal_cmd_root) / context_name, internal_wf_path)

    @staticmethod
    @lru_cache(maxsize=1)  # Only need to cache one result since core commands are fixed
    def _discover_core_commands(internal_wf_path: str) -> set[str]:
        """
        Discover core commands from the internal workflow.
        This is cached to avoid repeated filesystem scanning.
        
        Args:
            internal_wf_path: Path to the internal workflow
            
        Returns:
            set[str]: Set of discovered core command names
        """
        internal_cmd_root = os.path.join(internal_wf_path, "_commands")
        
        # --------------------------------------------------------
        # Deep-scan the internal workflow's _commands
        # directory to build the union of all command names.
        # --------------------------------------------------------
        discovered_commands: set[str] = set()
        for root, _dirs, files in os.walk(internal_cmd_root):
            if root.endswith('ErrorCorrection'):
                continue

            for filename in files:
                if (
                    filename.endswith(".py")
                    and not filename.startswith("_")
                    and filename != "__init__.py"
                ):
                    stem = os.path.splitext(filename)[0]

                    # Build qualified name relative to _commands root.
                    rel_path = os.path.relpath(os.path.join(root, filename), internal_cmd_root)
                    parts = os.path.splitext(rel_path)[0].split(os.sep)
                    qualified_name = "/".join(parts) if len(parts) > 1 else parts[0]
                    discovered_commands.add(qualified_name)

        if not discovered_commands:
            raise RuntimeError("No core commands discovered in internal workflow: {internal_wf_path}")

        return discovered_commands

    @classmethod
    def _register_context_class(cls, command_directory: "CommandDirectory", context_name: str, context_dir: Path, workflow_folderpath: str):
        """Register a context callback implementation if a file named '_<ContextName>.py' (or without .py) exists."""

        # Primary expected filename with .py extension
        candidate_with_ext = context_dir / f"_{context_name}.py"
        candidate_no_ext = context_dir / f"_{context_name}"

        context_file_path: Optional[Path] = None
        if candidate_with_ext.is_file():
            context_file_path = candidate_with_ext
        elif candidate_no_ext.is_file():
            context_file_path = candidate_no_ext

        if context_file_path is None:
            # No context callback implementation available – silently ignore
            return

        metadata = ContextMetadata(
            workflow_folderpath=workflow_folderpath,
            context_module_path=str(context_file_path),
            context_class="Context",
        )
        command_directory.register_context_metadata(context_name, metadata)

    # ---------------------- Context Metadata helpers ---------------------- #

    def register_context_metadata(self, context_name: str, metadata: "ContextMetadata"):
        if not (context_name and metadata):
            raise ValueError("context_name and metadata are required")
        self.map_context_2_metadata[context_name] = metadata

    def get_context_metadata(self, context_name: str) -> "ContextMetadata":
        if context_name not in self.map_context_2_metadata:
            raise KeyError(f"Context '{context_name}' not found in map_context_2_metadata")
        return self.map_context_2_metadata[context_name]

    # ---------------------- Validators for context metadata ---------------------- #

    @field_validator("map_context_2_metadata", mode="before")
    def validate_map_context_2_metadata(cls, v, info: ValidationInfo):
        if not isinstance(v, dict):
            raise ValueError("map_context_2_metadata must be a dictionary.")

        workflow_folderpath = info.data.get('workflow_folderpath')

        for context_key, metadata_val in v.items():
            if not context_key:
                raise ValueError("Context key cannot be empty.")
            if isinstance(metadata_val, dict):
                if 'workflow_folderpath' not in metadata_val and workflow_folderpath:
                    metadata_val['workflow_folderpath'] = workflow_folderpath
                v[context_key] = ContextMetadata(**metadata_val)
            elif not isinstance(metadata_val, ContextMetadata):
                raise ValueError(f"Invalid metadata for context key '{context_key}'")
        return v

    # ------------------------------------------------------------------
    # Lazy hydration helpers
    # ------------------------------------------------------------------

    def ensure_command_hydrated(self, command_name: str):
        """Populate Signature-related metadata & utterances on demand.

        This method is idempotent and inexpensive after the first call
        thanks to the @lru_cache on the internal helper.
        """

        command_metadata = self.map_command_2_metadata.get(command_name)
        if command_metadata is None:
            raise KeyError(f"Command '{command_name}' is not registered. command_metadata is missing")

        utterance_metadata = self.map_command_2_utterance_metadata.get(command_name)
        if utterance_metadata is None:
            utterance_metadata = UtteranceMetadata(
                workflow_folderpath = '',
                plain_utterances = [],
                template_utterances = [],
                generated_utterances_module_filepath = 'dummy',
                generated_utterances_func_name = 'dummy'               
            )

        # If already hydrated in this process we are done. (When a directory is
        # deserialized from `command_directory.json` in a fresh process, the
        # string fields on `command_metadata` are already populated from the
        # snapshot, so we still ensure the command's in-memory `from_` edges
        # have been resolved once for its `Signature` class before returning.)
        if command_metadata.response_generation_class_name is not None and (
            command_metadata.input_for_param_extraction_class is not None and \
            utterance_metadata.plain_utterances
        ):
            _ensure_workflow_field_sources_resolved(command_metadata, self, command_name)
            return

        _lazy_hydrate_metadata(
            command_metadata.response_generation_module_path,
            command_metadata,
            self,
            command_name,
        )


# ------------------------------------------------------
# Module-level helper with its own cache so that repeated
# hydration calls across directories are cheap.
# ------------------------------------------------------

# Guard against re-entrant `resolve_field_sources` calls when `resolve_field_sources`
# loads `CommandContextModel`, which builds `CommandDirectory` and hydrates commands.
_RESOLVING_FIELD_SOURCES_WORKFLOWS: set[str] = set()
# Cache of `(workflow_abs_path, command_name) -> Signature class` already resolved
# and logged in this process. Keying on the `Signature` class object ensures that
# resolution and error logging happen at most once per `(workflow, command, Signature)`
# while a reloaded module (producing a new `Signature` class object) is resolved again.
_RESOLVED_COMMAND_SIGNATURES: dict[tuple[str, str], type] = {}
# ``(workflow_abs_path, command_name)`` already logged via :func:`_handle_command_field_source_errors`.
_LOGGED_FIELD_SOURCE_COMMAND_ERRORS: set[tuple[str, str]] = set()
_INTERNAL_CME_WORKFLOW_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_workflows", "command_metadata_extraction")
)


def clear_resolved_command_signatures_cache() -> None:
    """Clear the per-command ``(workflow, command) -> Signature`` resolution cache."""
    _RESOLVED_COMMAND_SIGNATURES.clear()
    _LOGGED_FIELD_SOURCE_COMMAND_ERRORS.clear()


def _is_subject_workflow_command(
    command_name: str, metadata: CommandMetadata, directory: CommandDirectory
) -> bool:
    """True when *command_name* belongs to a subject workflow rather than the internal CME workflow."""
    if command_name in directory.core_command_names:
        return False
    if os.path.abspath(metadata.workflow_folderpath or "") == _INTERNAL_CME_WORKFLOW_PATH:
        return False
    if os.path.abspath(directory.workflow_folderpath or "") == _INTERNAL_CME_WORKFLOW_PATH:
        return False
    return True


def _handle_command_field_source_errors(
    command_name: str,
    errs: list[str],
    *,
    workflow_folderpath: str | None = None,
) -> bool:
    """Policy for a command whose ``Signature`` has ``from_`` / ``available_from`` errors.

    Applies to execution and planning commands alike: log each problem once per process
    (keyed by workflow folder and command name), leave the offending field's
    ``available_from`` edge unwritten, and keep the command loadable (routable for
    execution commands; matchable for planning commands). Preview ``--check`` still
    fails on the reported errors.

    Returns ``True`` to keep hydrating or loading the command, or ``False`` to skip it.
    Kept in one place so the policy can be switched cleanly.
    """
    if errs:
        wf_abs = os.path.abspath(workflow_folderpath) if workflow_folderpath else ""
        log_key = (wf_abs, command_name)
        if log_key not in _LOGGED_FIELD_SOURCE_COMMAND_ERRORS:
            _LOGGED_FIELD_SOURCE_COMMAND_ERRORS.add(log_key)
            logger.error(f"command {command_name!r}: " + "; ".join(errs))
    return True


def _ensure_workflow_field_sources_resolved(
    metadata: CommandMetadata,
    directory: CommandDirectory,
    command_name: str,
    signature_cls: Optional[type] = None,
) -> bool:
    """Ensure ``resolve_field_sources`` has run for the workflow and the command's
    ``Signature`` has its ``available_from`` populated (including when ``CommandDirectory``
    was loaded from a persisted JSON snapshot in a new process).

    Cached per ``(workflow_folder, command_name, Signature class identity)`` so each
    command's resolution and error logging runs at most once per ``Signature`` class.
    Returns ``True`` when hydration should proceed, ``False`` if the command should be skipped.
    """
    if not _is_subject_workflow_command(command_name, metadata, directory):
        return True
    wf_folder = directory.workflow_folderpath or metadata.workflow_folderpath
    if not wf_folder:
        return True
    wf_abs = os.path.abspath(wf_folder)
    if wf_abs in _RESOLVING_FIELD_SOURCES_WORKFLOWS:
        return True

    module_path = metadata.response_generation_module_path
    if signature_cls is None:
        module = python_utils.get_module(module_path, metadata.workflow_folderpath or wf_folder)
        signature_cls = getattr(module, "Signature", None) if module else None
    if signature_cls is None:
        return True

    cache_key = (wf_abs, command_name)
    if _RESOLVED_COMMAND_SIGNATURES.get(cache_key) is signature_cls:
        return True

    from fastworkflow import planning_commands  # circular import: planning_commands -> command_context_model -> command_directory

    _RESOLVING_FIELD_SOURCES_WORKFLOWS.add(wf_abs)
    try:
        resolution = planning_commands.resolve_field_sources(wf_abs)
    finally:
        _RESOLVING_FIELD_SOURCES_WORKFLOWS.discard(wf_abs)

    cmd_problems = planning_commands.resolve_signature_field_sources(
        command_name, signature_cls, module_path, resolution
    )
    _RESOLVED_COMMAND_SIGNATURES[cache_key] = signature_cls

    errs = resolution.errors_for(command_name) or [
        p.format() for p in cmd_problems if p.severity == "error"
    ]
    # ``skipped_commands`` lists commands with a field-source error, not unloadable ones.
    if errs or command_name in resolution.skipped_commands:
        if not _handle_command_field_source_errors(
            command_name, errs, workflow_folderpath=wf_abs
        ):
            metadata.parameter_extraction_signature_module_path = None
            metadata.input_for_param_extraction_class = None
            metadata.command_parameters_class = None
            return False
    return True


# We purposely do *not* cache by arguments because `metadata`/`directory` are
# unhashable Pydantic objects.  The helper itself safeguards against running
# twice for the same command by mutating `metadata` in-place.
def _lazy_hydrate_metadata(module_path: str, metadata: CommandMetadata, directory: CommandDirectory, command_name: str):
    """Load the command's module once and enrich its metadata object.

    Note: `metadata` is mutated in place which is fine because the object
    is unique within its `CommandDirectory` instance and, through the
    outer LRU cache, this function will execute at most once per command.
    """

    module = python_utils.get_module(module_path, metadata.workflow_folderpath)
    if not module:
        return  # cannot hydrate

    signature_cls = getattr(module, "Signature", None)
    if not signature_cls:
        return

    if not _ensure_workflow_field_sources_resolved(
        metadata, directory, command_name, signature_cls=signature_cls
    ):
        return

    metadata.parameter_extraction_signature_module_path = module_path
    metadata.input_for_param_extraction_class = "Signature"
    if hasattr(signature_cls, "Input"):
        metadata.command_parameters_class = "Signature.Input"

    # Utterance metadata (if not already registered)
    if command_name not in directory.map_command_2_utterance_metadata:
        CommandDirectory._populate_utterance_metadata_for_command(
            directory,
            command_name,
            metadata,
            metadata.workflow_folderpath or directory.workflow_folderpath,
            module,
        )

# Files (beyond *.py) that the command/routing snapshots derive from. Changes
# to these must also invalidate the persisted caches.
_CONTEXT_MODEL_BASENAMES = ("context_inheritance_model.json",)


#: The load error for a workflow that still has a ``_skills/`` folder (see ``check_no_planning_commands_folder``).
PLANNING_COMMANDS_FOLDER_RETIRED = (
    "_skills/ is no longer read; move these files into _commands/ (first make sure any script "
    "that regenerates _commands/ preserves them; see docs/migrating_skills_to_commands.md)"
)


def check_no_planning_commands_folder(workflow_folderpath: str) -> None:
    """Raise ValueError when the workflow folder still has a retired ``_skills/`` folder.

    Planning commands are read from ``_commands/`` (as are execution commands), so a workflow
    that keeps its planning commands in ``_skills/`` must move them before it loads.
    """
    if os.path.isdir(os.path.join(workflow_folderpath, "_skills")):
        raise ValueError(PLANNING_COMMANDS_FOLDER_RETIRED)


def _command_kind_from_ast(command_file_path: Path) -> tuple[str, bool]:
    """Return ``(response_generation_class_name, is_planning)`` without importing the module."""
    try:
        source = command_file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(command_file_path))
    except SyntaxError:
        raise
    except OSError as exc:
        raise ValueError(f"{command_file_path}: could not read command source ({exc})") from exc
    has_response_generator = False
    has_plan_generator = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name == "ResponseGenerator":
                has_response_generator = True
            elif node.name == "PlanGenerator":
                has_plan_generator = True
    if has_response_generator and has_plan_generator:
        raise ValueError(
            f"{command_file_path}: defines both ResponseGenerator and PlanGenerator"
        )
    if has_plan_generator:
        return "PlanGenerator", True
    return "ResponseGenerator", False


def _command_source_roots(workflow_folderpath: str) -> list[Path]:
    """Return every `_commands` root that contributes to this workflow.

    This includes the workflow's own `_commands` plus any base-workflow
    `_commands` declared via `workflow_inheritance_model.json`. Base resolution
    is best-effort: if it fails we still fingerprint the local sources so the
    cache never silently goes stale.
    """
    roots: list[Path] = []
    with contextlib.suppress(Exception):
        inheritance_model = WorkflowInheritanceModel.load(workflow_folderpath)
        roots.extend(
            Path(base_path) / "_commands"
            for base_path in inheritance_model.resolve_base_paths(workflow_folderpath)
        )
    roots.append(Path(workflow_folderpath) / "_commands")
    # Core commands and their context classes are registered into every directory
    # (see `_register_core_commands`), so their source must invalidate it too.
    import fastworkflow
    roots.append(Path(fastworkflow.get_internal_workflow_path("command_metadata_extraction")) / "_commands")
    return roots


def compute_commands_source_fingerprint(workflow_folderpath: str) -> str:
    """Fingerprint the command sources a snapshot is derived from.

    The fingerprint is a hash over the *set* of source files and their
    `(size, mtime_ns)`. Unlike a "newest mtime" comparison, this detects
    additions, deletions, renames, and edits uniformly, which is required to
    safely invalidate `command_directory.json` and `routing_definition.json`.
    """
    workflow_folderpath = str(Path(workflow_folderpath).resolve())
    entries: list[tuple[str, int, int]] = []

    for root in _command_source_roots(workflow_folderpath):
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix == ".py" or p.name in _CONTEXT_MODEL_BASENAMES:
                st = p.stat()
                entries.append((str(p), st.st_size, st.st_mtime_ns))

    # No hierarchy file is fingerprinted: parents come from the live chain, and a leftover
    # context_hierarchy_model.json is ignored (see CommandContextModel.load).

    return hashlib.sha256(repr(sorted(entries)).encode("utf-8")).hexdigest()


def _recorded_modules_exist(directory: "CommandDirectory") -> bool:
    """Whether every absolute module path the snapshot recorded is still on disk.

    The fingerprint does not cover fastworkflow's own core commands, whose
    absolute paths the snapshot also records, so a moved or reinstalled
    fastworkflow leaves a snapshot that fingerprints clean but points at files
    that are gone. Not folded into the fingerprint because checkpoint namespaces
    are keyed on it too.
    """
    paths = [
        path
        for metadata in directory.map_command_2_metadata.values()
        for path in (
            metadata.parameter_extraction_signature_module_path,
            metadata.response_generation_module_path,
        )
    ]
    paths.extend(
        metadata.context_module_path for metadata in directory.map_context_2_metadata.values()
    )
    return all(os.path.isfile(path) for path in paths if path and os.path.isabs(path))


@lru_cache(maxsize=32)
def get_cached_command_directory(workflow_folderpath: str) -> CommandDirectory:
    """Return a cached CommandDirectory, rebuilding only when sources change.

    The cache key is the *resolved* workflow path so that duplicate relative
    paths share the same entry. We validate the persisted `command_directory.json`
    against a fingerprint of the live `_commands` tree (see
    `compute_commands_source_fingerprint`). When the fingerprint matches we
    deserialize the JSON directly; otherwise we rebuild via
    `CommandDirectory.load()` and persist the new snapshot. The fingerprint is
    deletion- and rename-aware, unlike a plain mtime comparison.
    """

    workflow_folderpath = str(Path(workflow_folderpath).resolve())
    # Checked before the fast path: a snapshot written before a _skills/ folder appeared still matches the fingerprint
    check_no_planning_commands_folder(workflow_folderpath)

    cache_folder = Path(CommandDirectory.get_commandinfo_folderpath(workflow_folderpath))
    cache_file = cache_folder / "command_directory.json"

    current_fingerprint = compute_commands_source_fingerprint(workflow_folderpath)

    # Fast-path: load JSON if its stamped fingerprint matches the live sources
    if cache_file.exists():
        with contextlib.suppress(Exception):
            cached = CommandDirectory.model_validate_json(cache_file.read_text())
            if cached.source_fingerprint == current_fingerprint and _recorded_modules_exist(cached):
                return cached

    # (Re)build and persist (save() re-stamps the fingerprint)
    directory = CommandDirectory.load(workflow_folderpath)
    directory.save()
    return directory
