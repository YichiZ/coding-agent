import anthropic
import os
from dotenv import load_dotenv

USER_MULTI_TURN_PROMPT="""
We are testing a loop. After each of your replies you will receive 'Continue'. Write the powers of two from 2 to 64, one per reply: the first reply is 2, the next is 4, and so on. Each reply contains only the number. In the reply that contains 64, put DONE alone on the line after it.
"""

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
      system="You are coding agent.",
      max_tokens=16_000,
      messages=messages,
      stop_sequences=["DONE"],
    )

    text = "\n".join(b.text for b in message.content if b.type =="text")
    print(text)

    # Loop exit conditions
    if message.stop_reason == "stop_sequence":   # Claude said DONE
      print("[Stopped: DONE]")
      break
    if message.stop_reason == "max_tokens":
      print("[Stopped: hit max_tokens, reply was cut off]")
      break
    if not text.strip():
      print("[Stopped: empty reply]")
      break
    
    # Append the response and continue the loop
    messages.append({"role": "assistant", "content": message.content})
    messages.append({"role": "user", "content": "Continue"})


if __name__ == "__main__":
  load_dotenv()
  main()
