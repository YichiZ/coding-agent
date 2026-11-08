import json
import os
import subprocess
from pathlib import Path

from anthropic import beta_tool

import anthropic
from anthropic.lib.tools import BetaFunctionTool
from dotenv import load_dotenv

SYSTEM_PROMPT = """
You are a coding agent.

You have access to a few tools.


Print out on a single line when you execute a tool.
"""

USER_MULTI_TURN_PROMPT = """
List all files. Summarize the python files
"""


def resolve_abs_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


@beta_tool
def read_file_tool(filename: str) -> dict[str, any]:
    """
    Read the content of a file.

    Args:
        filename: Path to a file.
    """
    full_path = resolve_abs_path(filename)
    print(full_path)
    with open(str(full_path), "r") as f:
        content = f.read()
    return {"file_path": str(full_path), "content": content}


@beta_tool
def edit_file_tool(path: str, old_str: str, new_str: str) -> dict[str, any]:
    """
    Edit a file inside a path. If the old_str is empty, create the file, otherwise
    overwrite it. If we have an old_str but we don't see it, return an error.

    Args:
        path: Path to the file.
        old_str: String to be replaced, or empty if the file does not exist.
        new_str: New string to replace or append.
    """
    full_path = resolve_abs_path(path)
    if old_str == "":
        full_path.write_text(new_str, encoding="utf-8")
        return {"path": str(full_path), "action": "created_file"}
    original = full_path.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        return {"path": str(full_path), "action": "old_str not found"}
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    return {"path": str(full_path), "action": "edited"}


@beta_tool
def list_files_tool(path: str) -> dict[str, any]:
    """
    List all the files and folders in a directory.

    Args:
        path: Path to a directory.
    """
    full_path = resolve_abs_path(path)
    all_files = []
    for item in full_path.iterdir():
        all_files.append(
            {"filename": item.name, "type": "file" if item.is_file() else "dir"}
        )
    return {"path": str(full_path), "files": all_files}


@beta_tool
def bash_tool(command: str) -> str:
    """
    Runs a shell command and returns its combined stdout and stderr. Use this to inspect files, run scripts, and check command output.

    Args:
        command: The shell command to run.
    """
    r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return (r.stdout + r.stderr).strip() or "(no output)"


TOOL_REGISTRY: dict[str, BetaFunctionTool] = {
    "bash_tool": bash_tool,
    "read_file_tool": read_file_tool,
    "edit_file_tool": edit_file_tool,
    "list_files_tool": list_files_tool,
}

TOOLS = [toolFn.to_dict() for toolFn in TOOL_REGISTRY.values()]


def main() -> None:
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    messages = [
        {
            "role": "user",
            "content": USER_MULTI_TURN_PROMPT,
        }
    ]
    # Cap iterations to 20 (we should not reach that)
    for i in range(20):
        message = client.messages.create(
            model="claude-sonnet-5-5",
            system=SYSTEM_PROMPT,
            max_tokens=2000,
            messages=messages,
            stop_sequences=["DONE"],
            tools=TOOLS,
        )

        # Loop exit conditions
        if message.stop_reason == "stop_sequence":  # Claude said DONE
            print("[Stopped: DONE]")
            break
        if message.stop_reason == "max_tokens":
            print("[Stopped: hit max_tokens, reply was cut off]")
            break
        if message.stop_reason == "end_turn":
            print("[Stopped: turn ended]")
            break

        # check tool uses
        tool_invocations = [
            (block.name, block.input, block.id)
            for block in message.content
            if block.type == "tool_use"
        ]

        messages.append({"role": "assistant", "content": message.content})

        # No tool use = end
        if not tool_invocations:
            break
        else:
            tool_uses = []
            for name, args, id in tool_invocations:
                tool = TOOL_REGISTRY[name]

                print(name, args)
                resp = tool(**args)         

                (
                    tool_uses.append(
                        {"type": "tool_result", "tool_use_id": id, "content": json.dumps(resp)}
                    ),
                )

            messages.append({"role": "user", "content": tool_uses})


if __name__ == "__main__":
    load_dotenv()
    main()
