# FastWorkflow Build Tool CLI Specification

## Overview
This document specifies the command-line interface (CLI) for the FastWorkflow build tool. The CLI enables users to generate FastWorkflow command files and context models for any Python application by analyzing its source code and outputting the generated files to a specified directory.

## Use Cases
- Generate FastWorkflow command files and a context model for a Python application.
- Generate global commands for top-level functions in the application.
- Specify source and output directories.
- Optionally preview actions (dry run), enable verbose logging, specify a custom context model name, and control file overwriting.

## Arguments

| Argument         | Short | Type    | Required | Description                                      |
|------------------|-------|---------|----------|--------------------------------------------------|
| --source-dir     | -s    | string  | Yes      | Source code directory to analyze                 |
| --output-dir     | -o    | string  | Yes      | Where to place generated files                   |
| --dry-run        |       | flag    | No       | Do not write files, just print actions           |
| --verbose        | -v    | flag    | No       | Print detailed logs                              |
| --context-name   |       | string  | No       | Name for the context model JSON                  |
| --overwrite      |       | flag    | No       | Overwrite files in output directory if present   |

## Argument Details
- **--source-dir, -s**: Path to the target application's source directory. Must exist and be readable.
- **--output-dir, -o**: Path to the output directory for generated files. Must exist and be writable.
- **--dry-run**: If set, the tool will not write any files but will print the actions it would take.
- **--verbose, -v**: If set, the tool will print detailed logs for debugging and transparency.
- **--context-name**: If provided, specifies the name of the generated context model JSON file. Must be a valid filename.
- **--overwrite**: If set, the tool will overwrite existing files in the output directory without prompting.

## Validation Rules
- All required arguments must be provided.
- Directories must exist and be accessible.
- If `--overwrite` is not set, prompt before overwriting files.

## Usage Examples

**Basic usage:**
```sh
python -m fastworkflow.build --source-dir my_app/ --output-dir my_app/build/
```

**With optional arguments:**
```sh
python -m fastworkflow.build -s my_app/ -o my_app/build/ --dry-run --verbose
python -m fastworkflow.build -s my_app/ -o my_app/build/ --context-name my_context.json --overwrite
```

**Help text:**
```sh
python -m fastworkflow.build --help
```

## Help Text Example
```
usage: fastworkflow.build [-h] --source-dir SOURCE_DIR --output-dir OUTPUT_DIR [--dry-run] [--verbose] [--context-name CONTEXT_NAME] [--overwrite]

Generate FastWorkflow command files and context model from a Python application.

optional arguments:
  -h, --help            show this help message and exit
  --source-dir SOURCE_DIR, -s SOURCE_DIR
                        Path to the source directory of the target application
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Path to save the generated command files and context model
  --dry-run             Do not write files, just print actions
  --verbose, -v         Print detailed logs
  --context-name CONTEXT_NAME
                        Name for the context model JSON
  --overwrite           Overwrite files in output directory if present
```

## Review and Approval
- The above specification should be reviewed with stakeholders and updated as needed before implementation. 