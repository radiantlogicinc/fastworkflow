from enum import Enum


class CommandSource(str, Enum):
    BASE_COMMANDS = ("_base_commands",)
    COMMANDS = "_commands"
