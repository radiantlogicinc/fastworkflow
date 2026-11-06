"""
Main CLI entry point for fastWorkflow.
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import importlib.resources
from rich import print as rprint
from rich.live import Live
from rich.spinner import Spinner

_examples_dir_cache = None

def find_examples_dir():
    """Finds the bundled examples directory using importlib.resources."""
    global _examples_dir_cache
    
    # Return cached result if available
    if _examples_dir_cache is not None:
        return _examples_dir_cache
        
    with contextlib.suppress(ModuleNotFoundError, FileNotFoundError):
        # Use files() for robust path handling with importlib.resources
        resources_path = importlib.resources.files('fastworkflow')
        examples_path = resources_path / 'examples'
        if examples_path.is_dir():
            _examples_dir_cache = (examples_path, True)  # True indicates package examples
            return _examples_dir_cache
    
    # If not found in the package, look in the project root
    project_root = Path.cwd()
    examples_path = project_root / 'examples'
    
    if examples_path.is_dir():
        _examples_dir_cache = (examples_path, False)
    else:
        _examples_dir_cache = (None, False)
        
    return _examples_dir_cache

def list_examples(args):
    """List available bundled examples."""
    # Create a spinner for searching
    spinner = Spinner("dots", text="[bold green]Searching for examples...[/bold green]")
    
    # Use Live display for the spinner
    with Live(spinner, refresh_per_second=10):
        # Add a small delay to show the spinner (the actual search is very fast if cached)
        time.sleep(0.3)
        examples_dir, is_package = find_examples_dir()

    if not examples_dir:
        rprint("[bold red]Error:[/bold red] Could not find the bundled 'examples' directory.")
        sys.exit(1)

    rprint("\n[bold]Available examples:[/bold]")
    for item in sorted(examples_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('_'):
            rprint(f"- {item.name}")

def fetch_example(args):
    """Fetch a bundled example and copy it to the local filesystem."""
    # Create a spinner for searching
    spinner = Spinner("dots", text="[bold green]Fetching example...[/bold green]")
    
    # Use Live display for the spinner
    with Live(spinner, refresh_per_second=10):
        examples_dir, is_package = find_examples_dir()

    if not examples_dir:
        rprint("[bold red]Error:[/bold red] Could not find the bundled 'examples' directory.")
        sys.exit(1)

    source_path = examples_dir / args.name
    if not source_path.is_dir():
        rprint(f"[bold red]Error:[/bold red] Example '{args.name}' not found.")
        rprint("Use 'fastworkflow examples list' to see available examples.")
        sys.exit(1)

    # If examples are only found locally (not in package), skip the fetch operation
    if not is_package:
        rprint(f"Note: Example '{args.name}' is already in the local examples directory.")
        return source_path

    target_root = Path("./examples")
    target_path = target_root / args.name

    if target_path.exists() and not getattr(args, 'force', False):
        # Ask user for confirmation before overwriting
        response = input(f"Target directory '{target_path}' already exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            rprint("Operation cancelled.")
            sys.exit(0)
        
    target_root.mkdir(exist_ok=True)

    # Ignore generated files during copy
    ignore_patterns = shutil.ignore_patterns('___command_info', '__pycache__', '*.pyc')

    try:
        # Create a spinner for copying
        copy_spinner = Spinner("dots", text="[bold green]Copying files...[/bold green]")
        
        # Use Live display for the spinner
        with Live(copy_spinner, refresh_per_second=10):
            # Copy the example directory
            shutil.copytree(source_path, target_path, ignore=ignore_patterns, dirs_exist_ok=True)
            
            # Also copy the environment files from the examples directory if they don't exist locally
            env_file = examples_dir / "fastworkflow.env"
            passwords_file = examples_dir / "fastworkflow.passwords.env"
            
            local_env_file = target_root / "fastworkflow.env"
            local_passwords_file = target_root / "fastworkflow.passwords.env"
            
            # Check if env file exists locally before copying
            if env_file.exists() and not local_env_file.exists():
                shutil.copy2(env_file, local_env_file)
                time.sleep(0.5)  # Small delay to show the spinner
            
            # Check if passwords file exists locally before copying
            if passwords_file.exists():
                if not local_passwords_file.exists():
                    shutil.copy2(passwords_file, local_passwords_file)
                    time.sleep(0.5)  # Small delay to show the spinner
            elif not local_passwords_file.exists():
                # Create a template passwords file if the original doesn't exist and local one doesn't exist
                with open(local_passwords_file, "w") as f:
                    f.write("# Add your API keys below\n")
                    f.write("LITELLM_API_KEY_SYNDATA_GEN=<API KEY for synthetic data generation model>\n")
                    f.write("LITELLM_API_KEY_PARAM_EXTRACTION=<API KEY for parameter extraction model>\n")
                    f.write("LITELLM_API_KEY_PLANNER=<API KEY for the agent's task planner model>\n")
                    f.write("LITELLM_API_KEY_AGENT=<API KEY for the agent model>\n")
        
        # After copying, show the results
        if env_file.exists():
            if not local_env_file.exists():
                rprint(f"✅ Copied environment file to '{local_env_file}'")
            else:
                rprint(f"⚠️ Environment file already exists at '{local_env_file}', skipping copy")
            
            # Remind users to add their API keys
            rprint("\n[bold]NOTE:[/bold] You need to add your API keys to the passwords file before training.")
            rprint(f"      Edit '{local_passwords_file}' with your API keys.")
        else:
            rprint(f"⚠️ [yellow]Warning:[/yellow] Environment file not found at '{env_file}'")
        
        # Check if passwords file exists locally before copying
        if passwords_file.exists():
            if not local_passwords_file.exists():
                rprint(f"✅ Copied passwords file to '{local_passwords_file}'")
            else:
                rprint(f"⚠️ Passwords file already exists at '{local_passwords_file}', skipping copy")
        else:
            rprint(f"⚠️ [yellow]Warning:[/yellow] Passwords file not found at '{passwords_file}'")
            
            # Create a template passwords file if the original doesn't exist and local one doesn't exist
            if not local_passwords_file.exists():
                rprint(f"✅ Created template passwords file at '{local_passwords_file}'")
            else:
                rprint(f"⚠️ Using existing passwords file at '{local_passwords_file}'")
        
        rprint(f"\n✅ Example '{args.name}' copied to '{target_path}'")
        return target_path
    except Exception as e:
        rprint(f"[bold red]Error copying example:[/bold red] {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def find_default_env_files(workflow_path):
    """Find the appropriate default env files based on context.
    
    If the workflow path is within the examples directory, use the local examples env files.
    Otherwise, use local .env files in the current directory.
    
    Args:
        workflow_path: Optional path to the workflow directory
        
    Returns:
        tuple: (env_file_path, passwords_file_path)
    """
    # Resolve the workflow path to absolute path to handle relative paths correctly
    workflow_path = Path(workflow_path).resolve()        
    return workflow_path / "fastworkflow.env", workflow_path / "fastworkflow.passwords.env"

def add_build_parser(subparsers):
    """Add subparser for the 'build' command."""
    parser_build = subparsers.add_parser(
        "build",
        help="Generate FastWorkflow command files and context model from a Python application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_build.add_argument('--app-dir', '-s', required=True, help='Path to the source code directory of the application')
    parser_build.add_argument('--workflow-folderpath', '-w', required=True, help='Path to the workflow folder where commands will be generated')
    parser_build.add_argument('--overwrite', action='store_true', help='Overwrite files in output directory if present')
    parser_build.add_argument('--stub-commands', help='Comma-separated list of command names to generate stubs for')
    parser_build.add_argument('--no-startup', action='store_true', help='Skip generating the startup.py file')

    # Lazy-import build_main only if the user actually invokes the command
    def _build_main_wrapper(args):
        from .build.__main__ import build_main as _build_main
        return _build_main(args)

    parser_build.set_defaults(func=_build_main_wrapper)

def add_refine_parser(subparsers):
    """Add subparser for the 'refine' command."""
    parser_refine = subparsers.add_parser(
        "refine",
        help="Refine generated commands by enhancing metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_refine.add_argument('--workflow-folderpath', '-w', required=True, help='Path to the workflow folder to refine')

    # Lazy-import refine_main only if the user actually invokes the command
    def _refine_main_wrapper(args):
        from .refine.__main__ import refine_main as _refine_main
        return _refine_main(args)

    parser_refine.set_defaults(func=_refine_main_wrapper)

def add_train_parser(subparsers):
    """Add subparser for the 'train' command."""
    parser_train = subparsers.add_parser("train", help="Train the intent detection pipeline for a workflow.")
    parser_train.add_argument("workflow_folderpath", help="Path to the workflow folder")
    
    # Default env files will be determined at runtime based on the workflow path
    parser_train.add_argument(
        "env_file_path",
        nargs='?',
        default=None,
        help="Path to the environment file (default: .env in current directory, or bundled env file for examples)",
    )
    parser_train.add_argument(
        "passwords_file_path",
        nargs='?',
        default=None,
        help="Path to the passwords file (default: passwords.env in current directory, or bundled env file for examples)",
    )
    parser_train.add_argument(
        "--regenerate-utterances",
        action="store_true",
        help="Ignore the generated-utterance and DSPy parameter-example caches and call "
             "the LLM again. Costs money "
             "and breaks reproducibility against previous runs; use after editing seed "
             "utterances or when you suspect the cache is wrong.",
    )
    parser_train.set_defaults(func=lambda args: train_with_defaults(args))


def add_run_parser(subparsers):
    """Add subparser for the 'run' command (alias: 'run_cli').

    The alias names the transport the way its siblings do — run_cli /
    run_chatbot / run_fastapi_mcp — while 'run' stays valid for existing
    scripts and docs.
    """
    parser_run = subparsers.add_parser(
        "run",
        aliases=["run_cli"],
        help="Run a workflow's interactive assistant in this terminal.",
    )
    parser_run.add_argument("workflow_path", help="Path to the workflow folder")
    
    # Default env files will be determined at runtime based on the workflow path
    parser_run.add_argument(
        "env_file_path",
        nargs='?',
        default=None,
        help="Path to the environment file (default: .env in current directory, or bundled env file for examples)",
    )
    parser_run.add_argument(
        "passwords_file_path",
        nargs='?',
        default=None,
        help="Path to the passwords file (default: passwords.env in current directory, or bundled env file for examples)",
    )
    parser_run.add_argument("--context_file_path", help="Optional context file path", default="")
    parser_run.add_argument("--startup_command", help="Optional startup command", default="")
    parser_run.add_argument("--startup_action", help="Optional startup action", default="")
    parser_run.add_argument("--keep_alive", help="Optional keep_alive", default=True)
    parser_run.add_argument("--project_folderpath", help="Optional path to project folder containing application code", default=None)
    parser_run.add_argument(
        "--assistant", action="store_true", default=False,
        help="Run in assistant (non-agentic) mode. Default is agentic mode.",
    )
    parser_run.add_argument(
        "--generate_insights", action="store_true", default=False,
        help="Enable insights-distillation mode: compare teacher and student agents, extract planning and execution insights on divergence.",
    )
    parser_run.set_defaults(func=lambda args: run_with_defaults(args))

def add_skill_parser(subparsers):
    """Add the 'skill' command group: 'skill preview' renders a workflow's skills to markdown."""
    parser_skill = subparsers.add_parser("skill", help="Work with a workflow's skills (_skills/<name>.py).")
    skill_subparsers = parser_skill.add_subparsers(dest="action", required=True)
    parser_preview = skill_subparsers.add_parser(
        "preview",
        help="Write one markdown preview per skill (inputs, phases, facts, steps for sample inputs).",
    )
    parser_preview.add_argument("workflow_folder", help="Path to the workflow folder")
    parser_preview.add_argument(
        "--out", default=None,
        help="Folder for the previews (default: docs/skill_previews next to the workflow folder)",
    )
    parser_preview.add_argument(
        "--check", action="store_true",
        help="Write nothing; exit 1 when a preview is stale, missing or extra",
    )

    # Lazy-import the preview generator only if the user actually invokes the command
    def _preview_main_wrapper(args):
        from .skill_preview import preview_main
        code = preview_main(args)
        if code:
            sys.exit(code)

    parser_preview.set_defaults(func=_preview_main_wrapper)

def add_run_chatbot_parser(subparsers):
    """Add subparser for the 'run_chatbot' command (browser chat + observability viewer)."""
    parser_run_chatbot = subparsers.add_parser(
        "run_chatbot",
        help="Run a workflow's browser chatbot: a localhost UI to chat with the "
             "workflow and inspect what each turn did (observability viewer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Test mode [R19 as amended]: the workflow's FastAPI server is spawned
    # AUTOMATICALLY (loopback-only, CORS pinned to loopback origins, stopped
    # when the chatbot exits) so the chat works out of the box. Passing
    # --server-port skips the spawn and means an existing server on that port.
    parser_run_chatbot.add_argument(
        "--start-server", action="store_true", default=False,
        help=argparse.SUPPRESS,  # historical opt-in; auto-start is now the default
    )
    parser_run_chatbot.add_argument(
        "--server-port", type=int, default=None,
        help="Do not spawn a FastAPI server; use one already running on this "
             "port (Advanced panel). Omit to spawn a loopback server "
             "automatically (prefers port 8000).",
    )
    parser_run_chatbot.add_argument(
        "--allow-unsigned-jwt", action="store_true", default=False,
        help=argparse.SUPPRESS,  # historical opt-in; unsigned dev JWTs are now the
                                 # default for the loopback-only auto-spawned server
    )
    parser_run_chatbot.add_argument(
        "--expect-encrypted-jwt", action="store_true", default=False,
        dest="expect_encrypted_jwt",
        help="Spawn the server with JWT signature verification enabled (the "
             "chat tab then needs a signed token pasted into its Advanced "
             "connection panel)",
    )
    def _run_chatbot_main_wrapper(args):
        # Lazy import: the chatbot server is stdlib-only [R23] but pulls in the
        # fastworkflow package for state paths / the store. Env-file
        # auto-detection happens inside run_chatbot_main (per selected
        # workflow — the workflow may be chosen in the UI, after launch).
        from .run_chatbot.server import run_chatbot_main as _run_chatbot_main
        sys.exit(_run_chatbot_main(args))

    parser_run_chatbot.set_defaults(func=_run_chatbot_main_wrapper)

def add_distill_parser(subparsers):
    """Add subparser for the 'distill' command."""
    parser_distill = subparsers.add_parser(
        "distill",
        help="Compare a student run against a recorded teacher run: extract "
             "insights and record the divergences as annotations.",
        description="Distil two recorded runs of the same workflow. Both are read "
                    "from the observability store, so this re-runs as often as you "
                    "like over the same pair without re-running either agent — which "
                    "is what makes tuning the judge affordable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_distill.add_argument("workflow_folderpath", help="Path to the workflow folder")
    parser_distill.add_argument(
        "--teacher", metavar="EXPERIMENT_ID",
        help="Experiment id of the run to learn FROM (with --student)",
    )
    parser_distill.add_argument(
        "--student", metavar="EXPERIMENT_ID",
        help="Experiment id of the run to learn ABOUT (with --teacher)",
    )
    parser_distill.add_argument(
        "--self", dest="self_experiment", metavar="EXPERIMENT_ID",
        help="Self-distillation: one run with several trials per task; each failing "
             "trial is compared with a passing trial of the same task. Use instead "
             "of --teacher/--student.",
    )
    parser_distill.add_argument(
        "--pool-students", nargs="+", metavar="EXPERIMENT_ID",
        help="Pool distillation: runs of the model to improve. Every failing trial "
             "is compared with a passing reference - a passing trial of the same "
             "model if any, else a passing trial from --pool-teachers.",
    )
    parser_distill.add_argument(
        "--pool-teachers", nargs="*", default=[], metavar="EXPERIMENT_ID",
        help="Pool distillation: teacher runs; only their PASSING trials are used, "
             "and only for tasks with no passing student trial.",
    )
    parser_distill.add_argument(
        "--max-pairs-per-task", type=int, default=None, metavar="N",
        help="Self/pool distillation: distil at most N failing trials per task.",
    )
    parser_distill.add_argument(
        "--provenance", metavar="FILE",
        help="Write a JSON file mapping every compared pair to its runs, its "
             "reference source and the insights it produced.",
    )
    parser_distill.add_argument(
        "--teacher-attempt", type=int, default=None, metavar="N",
        help="--teacher/--student: the teacher attempt used as the reference for "
             "every task. Default: the lowest recorded attempt.",
    )
    parser_distill.add_argument(
        "--student-attempt", action="append", dest="student_attempts", metavar="N|all",
        help="--teacher/--student: a student attempt to compare with the reference, "
             "for every task; repeatable, or 'all' for every recorded attempt. "
             "Default: the lowest recorded attempt. Pairs are taken as given, "
             "whatever their outcomes.",
    )
    parser_distill.add_argument(
        "--task", action="append", dest="task_ids", metavar="TASK_ID",
        help="Only this task; repeatable. Default: every task the runs share.",
    )
    parser_distill.add_argument(
        "env_file_path", nargs="?", default=None,
        help="Path to the environment file (default: resolved from the workflow path)",
    )
    parser_distill.add_argument(
        "passwords_file_path", nargs="?", default=None,
        help="Path to the passwords file (default: resolved from the workflow path)",
    )
    parser_distill.set_defaults(func=_distill_main)


def _distill_mode_error(args) -> Optional[str]:
    """Why the distill mode flags do not name exactly one mode, or None."""
    modes = [
        bool(args.self_experiment),
        bool(args.pool_students),
        bool(args.teacher or args.student),
    ]
    if sum(modes) != 1 or (modes[2] and not (args.teacher and args.student)):
        return (
            "give exactly one of: --teacher EXPERIMENT_ID with --student "
            "EXPERIMENT_ID; --self EXPERIMENT_ID; or --pool-students EXPERIMENT_ID... "
            "[--pool-teachers EXPERIMENT_ID...]"
        )
    if args.pool_teachers and not args.pool_students:
        return "--pool-teachers needs --pool-students"
    teacher_mode = bool(args.teacher and args.student)
    if not teacher_mode and (args.teacher_attempt is not None or args.student_attempts):
        return "--teacher-attempt and --student-attempt need --teacher and --student"
    if teacher_mode and args.max_pairs_per_task is not None:
        return ("--max-pairs-per-task applies to --self and --pool-students; with "
                "--teacher/--student choose the pairs with --student-attempt")
    try:
        _student_attempt_choice(args.student_attempts)
    except ValueError as exc:
        return str(exc)
    return None


def _student_attempt_choice(values):
    """``--student-attempt`` values as compare_runs takes them: None, "all" or ints."""
    if not values:
        return None
    if any(str(v).strip().lower() == "all" for v in values):
        if len(values) > 1:
            raise ValueError("--student-attempt all cannot be combined with attempt numbers")
        return "all"
    try:
        return [int(v) for v in values]
    except ValueError:
        raise ValueError(
            f"--student-attempt takes attempt numbers or 'all', got {values!r}") from None


def _open_workflow_store(args, usage: str):
    """Load the workflow's env files and open its observability store.

    Returns the store, or None after printing why it could not.
    """
    import fastworkflow
    from dotenv import dotenv_values

    from fastworkflow.observability.store import get_observability_sink

    default_env, default_passwords = find_default_env_files(args.workflow_folderpath)
    if not os.path.exists(default_env):
        # The bundled examples keep one env file beside the workflow folders
        # rather than inside each one, so look up a level before giving up.
        parent = Path(args.workflow_folderpath).resolve().parent
        if (parent / "fastworkflow.env").exists():
            default_env = parent / "fastworkflow.env"
            default_passwords = parent / "fastworkflow.passwords.env"

    env_file = args.env_file_path or default_env
    passwords_file = args.passwords_file_path or default_passwords
    if not os.path.exists(env_file):
        print(
            f"Error: environment file not found at: {env_file}\n"
            f"Pass it explicitly: {usage}",
            file=sys.stderr,
        )
        return None

    # Model calls made here (the distillation judge, the policy readers in a live
    # replay) read their settings from the env, so it is loaded even though no
    # agent runs.
    fastworkflow.init(
        env_vars={**dotenv_values(env_file), **dotenv_values(passwords_file)}
    )

    # Recording has no switch: every caller gets a sink, and None means only that
    # the store could not be opened (the store has already logged why). Without a
    # store there is nothing to read and nothing to do.
    sink = get_observability_sink(args.workflow_folderpath)
    if sink is None:
        print(
            "Error: could not open the observability store for this workflow. "
            "This command reads recorded runs, so it needs that store.",
            file=sys.stderr,
        )
        return None
    return sink.store


def _distill_main(args):
    """Run a distillation comparison over recorded runs."""
    from fastworkflow import distillation_compare

    if problem := _distill_mode_error(args):
        print(f"Error: {problem}", file=sys.stderr)
        return 1
    store = _open_workflow_store(
        args,
        f"fastworkflow distill {args.workflow_folderpath} <env_file> <passwords_file> "
        "--teacher ... --student ...",
    )
    if store is None:
        return 1

    if args.pool_students:
        results = distillation_compare.compare_pool(
            store,
            student_experiments=args.pool_students,
            teacher_experiments=args.pool_teachers,
            workflow_folderpath=args.workflow_folderpath,
            task_ids=args.task_ids,
            max_pairs_per_task=args.max_pairs_per_task,
        )
    elif args.self_experiment:
        results = distillation_compare.compare_self(
            store,
            experiment_id=args.self_experiment,
            workflow_folderpath=args.workflow_folderpath,
            task_ids=args.task_ids,
            max_pairs_per_task=args.max_pairs_per_task,
        )
    else:
        results = distillation_compare.compare_runs(
            store,
            teacher_experiment_id=args.teacher,
            student_experiment_id=args.student,
            workflow_folderpath=args.workflow_folderpath,
            task_ids=args.task_ids,
            teacher_attempt=args.teacher_attempt,
            student_attempts=_student_attempt_choice(args.student_attempts),
        )
    if args.provenance:
        with open(args.provenance, "w", encoding="utf-8") as f:
            json.dump(
                distillation_compare.provenance_records(results),
                f,
                indent=1,
                ensure_ascii=False,
            )
    return _print_distill_report(results)


def _outcome_tag(passed) -> str:
    """The tag the report prints after an attempt: [pass], [fail], or nothing if ungraded."""
    return "" if passed is None else (" [pass]" if passed else " [fail]")


def _print_distill_report(results) -> int:
    """Print per-task findings. Returns a process exit code."""
    if not results:
        print("No tasks in common between the two runs; nothing to distil.")
        return 1

    diverged = insights = feedback = skipped = 0
    for result in results:
        if result.skipped_reason:
            skipped += 1
            print(f"  task {result.task_id}: skipped — {result.skipped_reason}")
            continue
        pair = (
            f"task {result.task_id} (attempt {result.student_attempt}"
            f"{_outcome_tag(result.student_passed)} vs {result.teacher_attempt}"
            f"{_outcome_tag(result.reference_passed)}"
            + (f", {result.reference_source} reference" if result.reference_source else "")
            + ")"
        )
        if not result.diverged:
            print(f"  {pair}: no divergence")
            continue
        diverged += 1
        insights += result.insight_count
        feedback += result.feedback_written
        print(f"  {pair}: DIVERGED")
        if result.execution_summary:
            for line in result.execution_summary.splitlines():
                print(f"      execution | {line}")
        if result.planning_summary:
            for line in result.planning_summary.splitlines()[:3]:
                print(f"      planning  | {line}")
        for insight in result.execution_insights + result.planning_insights:
            print(f"      insight   > {insight}")

    print(
        f"\n{len(results)} task(s): {diverged} diverged, {skipped} skipped, "
        f"{insights} insight(s), {feedback} annotation(s)"
    )
    return 0


def add_policy_parser(subparsers):
    """Add the 'policy' command group: 'policy replay' audits a table on recorded runs."""
    parser_policy = subparsers.add_parser(
        "policy",
        help="Work with a workflow's policy verifier table (_policy/table.py).",
    )
    policy_subparsers = parser_policy.add_subparsers(dest="action", required=True)
    parser_replay = policy_subparsers.add_parser(
        "replay",
        help="Replay a recorded experiment through a policy table and report every "
             "step it would block, next to what the recording shows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_replay.add_argument("workflow_folderpath", help="Path to the workflow folder")
    parser_replay.add_argument(
        "--experiment", required=True, metavar="EXPERIMENT_ID",
        help="Experiment id of the recorded run to replay",
    )
    parser_replay.add_argument(
        "--table", default=None, metavar="PATH[:ATTR]",
        help="Another table to audit (path.py or package.module; ATTR defaults to "
             "TABLE). Default: the workflow's own _policy/table.py",
    )
    parser_replay.add_argument(
        "--task", action="append", dest="task_ids", metavar="TASK_ID",
        help="Only this task; repeatable. Default: every task in the run.",
    )
    parser_replay.add_argument(
        "--live", action="store_true",
        help="Re-read every user message and judge every question with the policy "
             "model (LLM_POLICY_INTENT, else LLM_AGENT), as the live loop would",
    )
    parser_replay.add_argument(
        "env_file_path", nargs="?", default=None,
        help="Path to the environment file (default: resolved from the workflow path)",
    )
    parser_replay.add_argument(
        "passwords_file_path", nargs="?", default=None,
        help="Path to the passwords file (default: resolved from the workflow path)",
    )

    def _replay_main_wrapper(args):
        code = _policy_replay_main(args)
        if code:
            sys.exit(code)

    parser_replay.set_defaults(func=_replay_main_wrapper)


def _policy_replay_main(args) -> int:
    """Replay a recorded experiment through a policy table."""
    from fastworkflow.policy_loader import POLICY_FOLDER, TABLE_FILE, load_policy
    from fastworkflow.policy_replay import format_report, replay_experiment

    store = _open_workflow_store(
        args,
        f"fastworkflow policy replay {args.workflow_folderpath} <env_file> "
        "<passwords_file> --experiment ... [--table ...]",
    )
    if store is None:
        return 1
    point, facts = load_policy(args.workflow_folderpath, reference=args.table)
    if not point.enabled:
        print(
            f"{args.workflow_folderpath} has no {POLICY_FOLDER}/{TABLE_FILE}; "
            "pass --table to audit another table.",
            file=sys.stderr,
        )
        return 1
    report = replay_experiment(
        store, args.experiment, point.table, facts,
        task_ids=args.task_ids, live=args.live,
    )
    if not report.attempts:
        print(f"No attempts recorded for experiment {args.experiment}.", file=sys.stderr)
        return 1
    print(format_report(report))
    return 0


def add_run_fastapi_mcp_parser(subparsers):
    """Add subparser for the 'run_fastapi_mcp' command."""
    parser_run_fastapi_mcp = subparsers.add_parser("run_fastapi_mcp", help="Run a workflow as a FastAPI server with MCP support.")
    parser_run_fastapi_mcp.add_argument("workflow_path", help="Path to the workflow folder")
    
    # Default env files will be determined at runtime based on the workflow path
    parser_run_fastapi_mcp.add_argument(
        "env_file_path",
        nargs='?',
        default=None,
        help="Path to the environment file (default: .env in current directory, or bundled env file for examples)",
    )
    parser_run_fastapi_mcp.add_argument(
        "passwords_file_path",
        nargs='?',
        default=None,
        help="Path to the passwords file (default: passwords.env in current directory, or bundled env file for examples)",
    )
    parser_run_fastapi_mcp.add_argument("--context", help="Optional context (JSON string)", default=None)
    parser_run_fastapi_mcp.add_argument("--startup_command", help="Optional startup command", default=None)
    parser_run_fastapi_mcp.add_argument("--startup_action", help="Optional startup action (JSON string)", default=None)
    parser_run_fastapi_mcp.add_argument("--project_folderpath", help="Optional path to project folder containing application code", default=None)
    parser_run_fastapi_mcp.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on (default: 8000)")
    parser_run_fastapi_mcp.add_argument("--host", default="0.0.0.0", help="Host to bind the FastAPI server to (default: 0.0.0.0)")
    parser_run_fastapi_mcp.add_argument(
        "--cors_origin", default=None,
        help="Pin CORS to exactly this origin (no wildcard). Default: allow all origins.",
    )
    parser_run_fastapi_mcp.set_defaults(func=lambda args: run_fastapi_mcp_with_defaults(args))

def train_with_defaults(args):  # sourcery skip: extract-duplicate-method
    """Wrapper for train_main that sets default env file paths based on context."""
    if args.env_file_path is None or args.passwords_file_path is None:
        default_env, default_passwords = find_default_env_files(args.workflow_folderpath)
    if args.env_file_path is None:
        args.env_file_path = default_env
    if args.passwords_file_path is None:
        args.passwords_file_path = default_passwords

    # Check if the files exist and provide helpful error messages
    if not os.path.exists(args.env_file_path):
        print(f"Error: Environment file not found at: {args.env_file_path}", file=sys.stderr)
        # Check if this is an example workflow
        if "/examples/" in str(args.workflow_folderpath) or "\\examples\\" in str(args.workflow_folderpath):
            example_name = os.path.basename(args.workflow_folderpath)
            print("\nThis appears to be an example workflow. Please run:")
            print(f"  fastworkflow examples fetch {example_name}")
            print(f"  fastworkflow train ./examples/{example_name} ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env")
        else:
            print("\nPlease ensure this file exists with required environment variables.")
            print("You can create a basic .env file in your current directory.")
        sys.exit(1)

    if not os.path.exists(args.passwords_file_path):
        print(f"Error: Passwords file not found at: {args.passwords_file_path}", file=sys.stderr)
        # Check if this is an example workflow
        if "/examples/" in str(args.workflow_folderpath) or "\\examples\\" in str(args.workflow_folderpath):
            example_name = os.path.basename(args.workflow_folderpath)
            print("\nThis appears to be an example workflow. Please run:")
            print(f"  fastworkflow examples fetch {example_name}")
            print(f"  fastworkflow train ./examples/{example_name} ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env")
        else:
            print("\nPlease ensure this file exists with required API keys.")
            print("You can create a basic passwords.env file in your current directory.")
        sys.exit(1)

    # Lazy import here to avoid heavy startup when not needed
    from .train.__main__ import train_main as _train_main
    return _train_main(args)

def run_with_defaults(args):  # sourcery skip: extract-duplicate-method
    """Wrapper for run_main that sets default env file paths based on context."""
    if args.env_file_path is None or args.passwords_file_path is None:
        default_env, default_passwords = find_default_env_files(args.workflow_path)
    if args.env_file_path is None:
        args.env_file_path = default_env
    if args.passwords_file_path is None:
        args.passwords_file_path = default_passwords

    # Check if the files exist and provide helpful error messages
    if not os.path.exists(args.env_file_path):
        print(f"Error: Environment file not found at: {args.env_file_path}", file=sys.stderr)
        # Check if this is an example workflow
        if "/examples/" in str(args.workflow_path) or "\\examples\\" in str(args.workflow_path):
            example_name = os.path.basename(args.workflow_path)
            print("\nThis appears to be an example workflow. Please run:")
            print(f"  fastworkflow examples fetch {example_name}")
            print(f"  fastworkflow train ./examples/{example_name} ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env")
        else:
            print("\nPlease ensure this file exists with required environment variables.")
            print("You can create a basic .env file in your current directory.")
        sys.exit(1)

    if not os.path.exists(args.passwords_file_path):
        print(f"Error: Passwords file not found at: {args.passwords_file_path}", file=sys.stderr)
        # Check if this is an example workflow
        if "/examples/" in str(args.workflow_path) or "\\examples\\" in str(args.workflow_path):
            example_name = os.path.basename(args.workflow_path)
            print("\nThis appears to be an example workflow. Please run:")
            print(f"  fastworkflow examples fetch {example_name}")
            print(f"  fastworkflow train ./examples/{example_name} ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env")
        else:
            print("\nPlease ensure this file exists with required API keys.")
            print("You can create a basic passwords.env file in your current directory.")
        sys.exit(1)

    # Lazy import here to avoid heavy startup when not needed
    from .run.__main__ import run_main as _run_main
    return _run_main(args)

def _require_server_extra():
    """Ensure the optional `server` extra is installed before launching the FastAPI-MCP service.

    The FastAPI-MCP server (`fastworkflow run_fastapi_mcp`) is the only feature that
    needs the server-only dependencies (fastapi, uvicorn, fastapi-mcp, pyjwt). These
    live behind the optional `fastworkflow[server]` extra so that the core/client
    install (and `fastworkflow train`) stays lean. Fail early with a clear, actionable
    message instead of letting the subprocess die with a bare ImportError.
    """
    import importlib.util

    if missing := [
        pkg
        for pkg, mod in (
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("fastapi-mcp", "fastapi_mcp"),
            ("pyjwt", "jwt"),
        )
        if importlib.util.find_spec(mod) is None
    ]:
        rprint(
            "[bold red]Error:[/bold red] the FastAPI-MCP server requires the optional "
            "'server' extra, which is not installed.\n"
            f"Missing packages: {', '.join(missing)}\n\n"
            "Install it with one of:\n"
            "  pip install \"fastworkflow[server]\"\n"
            "  poetry install --extras server"
        )
        sys.exit(1)


def run_fastapi_mcp_with_defaults(args):  # sourcery skip: extract-duplicate-method
    """Wrapper for fastapi mcp server that sets default env file paths based on context."""
    _require_server_extra()
    if args.env_file_path is None or args.passwords_file_path is None:
        default_env, default_passwords = find_default_env_files(args.workflow_path)
    if args.env_file_path is None:
        args.env_file_path = default_env
    if args.passwords_file_path is None:
        args.passwords_file_path = default_passwords

    # Check if the files exist and provide helpful error messages
    if not os.path.exists(args.env_file_path):
        print(f"Error: Environment file not found at: {args.env_file_path}", file=sys.stderr)
        # Check if this is an example workflow
        if "/examples/" in str(args.workflow_path) or "\\examples\\" in str(args.workflow_path):
            example_name = os.path.basename(args.workflow_path)
            print("\nThis appears to be an example workflow. Please run:")
            print(f"  fastworkflow examples fetch {example_name}")
            print(f"  fastworkflow run_fastapi_mcp ./examples/{example_name} ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env")
        else:
            print("\nPlease ensure this file exists with required environment variables.")
            print("You can create a basic .env file in your current directory.")
        sys.exit(1)

    if not os.path.exists(args.passwords_file_path):
        print(f"Error: Passwords file not found at: {args.passwords_file_path}", file=sys.stderr)
        # Check if this is an example workflow
        if "/examples/" in str(args.workflow_path) or "\\examples\\" in str(args.workflow_path):
            example_name = os.path.basename(args.workflow_path)
            print("\nThis appears to be an example workflow. Please run:")
            print(f"  fastworkflow examples fetch {example_name}")
            print(f"  fastworkflow run_fastapi_mcp ./examples/{example_name} ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env")
        else:
            print("\nPlease ensure this file exists with required API keys.")
            print("You can create a basic passwords.env file in your current directory.")
        sys.exit(1)

    # Use subprocess to run python -m fastworkflow.run_fastapi_mcp with the correct arguments
    cmd = [
        sys.executable, '-m', 'fastworkflow.run_fastapi_mcp',
        '--workflow_path', args.workflow_path,
        '--env_file_path', args.env_file_path,
        '--passwords_file_path', args.passwords_file_path,
        '--port', str(args.port),
        '--host', args.host,
    ]
    if args.context:
        cmd.extend(['--context', args.context])
    if args.startup_command:
        cmd.extend(['--startup_command', args.startup_command])
    if args.startup_action:
        cmd.extend(['--startup_action', args.startup_action])
    if args.project_folderpath:
        cmd.extend(['--project_folderpath', args.project_folderpath])
    if getattr(args, 'cors_origin', None):
        cmd.extend(['--cors_origin', args.cors_origin])

    # Run the subprocess
    return subprocess.run(cmd).returncode

def main():
    """Main function for the fastworkflow CLI."""
    
    parser = argparse.ArgumentParser(
        description="fastWorkflow CLI tool for building, training, and running workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command help")

    # 'examples' command group
    parser_examples = subparsers.add_parser("examples", help="Manage bundled examples")
    examples_subparsers = parser_examples.add_subparsers(dest="action", required=True)

    # 'examples list' command
    parser_list = examples_subparsers.add_parser("list", help="List available examples")
    parser_list.set_defaults(func=list_examples)

    # 'examples fetch' command
    parser_fetch = examples_subparsers.add_parser("fetch", help="Fetch a specific example")
    parser_fetch.add_argument("name", help="The name of the example to fetch")
    parser_fetch.add_argument("--force", action="store_true", help="Force overwrite if example already exists")
    parser_fetch.set_defaults(func=fetch_example)
    
    # Add top-level commands
    add_build_parser(subparsers)
    add_refine_parser(subparsers)
    add_train_parser(subparsers)
    add_run_parser(subparsers)
    add_run_fastapi_mcp_parser(subparsers)
    add_run_chatbot_parser(subparsers)
    add_skill_parser(subparsers)
    add_distill_parser(subparsers)
    add_policy_parser(subparsers)

    try:
        args = parser.parse_args()
        args.func(args)
    except KeyboardInterrupt:
        rprint("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        rprint(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 