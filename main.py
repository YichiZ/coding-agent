import os
import subprocess

import anthropic
from dotenv import load_dotenv

SYSTEM_PROMPT = """
You are a coding agent.

You have access to a few tools


Print out on a single line when you execute a tool.
"""

USER_MULTI_TURN_PROMPT = """
List all files. Summarize the python files
"""

def bash_tool(command: str) -> str:
    """
    Runs a shell command and returns its combined output.
    :param command: The shell command to run.
    """
    r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return (r.stdout + r.stderr).strip() or "(no output)"


TOOLS = [
    {
        "name": "bash_tool",
        "description": "Runs a shell command and returns its combined stdout and stderr. Use this to inspect files, run scripts, and check command output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run (e.g., 'ls sandbox' or 'python main.py').",
                }
            },
            "required": ["command"],
        },
    },
]

TOOL_REGISTRY = {
    "bash_tool": bash_tool
}

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
        tool_invocations = [(block.name, block.input, block.id) for block in message.content if block.type == "tool_use"]

        messages.append({"role": "assistant", "content": message.content})

        # No tool use = end
        if not tool_invocations:
            break
        else:
            tool_uses = []
            for name, args, id in tool_invocations:
                tool = TOOL_REGISTRY[name]
                resp=""
                print(name, args)
                if name == "bash_tool":
                    resp = tool(args.get("command",""))

                tool_uses.append({
                    "type": "tool_result", 
                    "tool_use_id": id, 
                    "content": resp
                }),
            
            messages.append({
                "role": "user",
                "content": tool_uses
            })


if __name__ == "__main__":
    load_dotenv()
    main()
