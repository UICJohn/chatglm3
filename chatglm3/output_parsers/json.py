from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from typing import Any, Callable
import re
import json


def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: str) -> str:
    """
    The LLM response for `action_input` may be a multiline
    string containing unescaped newlines, tabs or quotes. This function
    replaces those characters with their escaped counterparts.
    (newlines in JSON must be double-escaped: `\\n`)
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    multiline_string = re.sub(
        r'("action_input"\:\s*")(.*)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )

    return multiline_string


def parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = json.loads
) -> dict:
    """
    Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    print("++++++++++++json_string: ", json_string)
    # Try to find JSON string within triple backticks
    pattern = r'[^`]*?({[^`]*?})'
    match = re.findall(pattern, json_string, re.DOTALL)
    # If no match found, assume the entire string is a JSON string
    # if match is None:
    #     json_str = json_string
    # else:
    #     # If match found, use the content within the backticks
    #     json_str = match.group(2)
    # print("json_str: ", json_str)
    if len(match) == 0:
      json_str = json_string
    else:
      json_str = match[0].strip()
      print("------------match here")

    print("===========json_string: ", json_string)
    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip()

    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    parsed = parser(json_str)

    return parsed


class JSONAgentOutputParser(AgentOutputParser):
  def parse(self, text: str) -> AgentAction | AgentFinish:
    try:
      # this will work IF the text is a valid JSON with action and action_input
      # log.info("Received text: ", text)
      response = parse_json_markdown(text)

      action, action_input = response["action"], response["action_input"]

      if action == "Final Answer":
        # this means the agent is finished so we call AgentFinish
        return AgentFinish({"output": action_input}, text)

      # otherwise the agent wants to use an action, so we call AgentAction
      return AgentAction(action, action_input, text)
    except Exception:
      # sometimes the agent will return a string that is not a valid JSON
      # often this happens when the agent is finished
      # so we just return the text as the output
      raise OutputParserException(f'Could not parse output: {text}', )
