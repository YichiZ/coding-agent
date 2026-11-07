import anthropic
import os
from dotenv import load_dotenv

# Add a loop
# Add tool calls
# Add a limit

def main() -> None:
  client = anthropic.Anthropic(api_key=os.environ.get(
    "ANTHROPIC_API_KEY"),
  )
  message = client.messages.create(
    model="claude-sonnet-5-5",
    system="You are coding agent.",
    max_tokens=1000,
    messages=[
      {
        "role": "user",
        "content": "Design a event-bus class in python",
      }
    ],
  )

  for block in message.content:
    if block.type == "text":
      print(block.text)


if __name__ == "__main__":
  load_dotenv()
  main()
