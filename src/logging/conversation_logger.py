import json
import os
from collections import defaultdict

from ._common import _now

def generate_conv_ids(start_id, num_simulations):
    """
    Generate conversation IDs starting from start_id.

    :param start_id: Starting conversation ID (e.g., "001", "042")
    :param num_simulations: Number of simulations to run
    :return: List of conversation IDs with proper zero-padding
    """
    if start_id.isdigit():
        start_num = int(start_id)
        width = len(start_id)

        conv_ids = []
        for i in range(num_simulations):
            conv_ids.append(str(start_num + i).zfill(width))
        return conv_ids
    else:
        return [f"{start_id}_{i+1}" for i in range(num_simulations)]

class ConversationLogger:
    """
    Minimal, append-only logger to track conversations and tool calls in a JSON file for research analysis.

    Attributes:
        filepath: filepath to the logging directory
        data: dictionary tracking conversation id, scenario parameters, dialogue, and tool calls
        turns: dictionary tracking each dialogue turn for the agent LLM and the user LLM
        tool calls: list tracking each tool call made by the user LLM
    """

    def __init__(self, log_dir, conversation_id, scenario_type, seed):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.conversation_id = conversation_id

        self.turns = {}
        self.tool_calls = []

        self.data = {
            "conversation_id": self.conversation_id,
            "scenario_type": scenario_type,
            "created timestamp": _now(),
            "seed": seed,
            "scenario_parameters": None,
            "summary": None,
            "dialogue": [],
            "tool_calls": [],
            "errors": [],
        }

        self._save()

    def setup(self, scenario_parameters):
        """
        Record the scenario parameters for this conversation.

        :param scenario_parameters: dictionary of scenario-specific parameters
        """
        self.data["scenario_parameters"] = scenario_parameters

    def _get_or_create_turn(self, turn_number):
        """
        Return the dialogue turn entry for turn_number, creating and
        registering it if it does not yet exist.

        :param turn_number: the dialogue turn number
        :return: the turn entry dict
        """
        if not self.turns.get(turn_number, None):
            entry = {
                "turn": turn_number,
                "user": None,
                "agent": None,
                "tools": [],
            }
            self.turns[turn_number] = entry
            self.data["dialogue"].append(entry)

        return self.turns[turn_number]

    def add_turn(self, turn_number, role, content):
        """
        Record a message for a dialogue turn.

        :param turn_number: the dialogue turn number
        :param role: either "user" or "agent"
        :param content: the text content of the message
        """
        turn = self._get_or_create_turn(turn_number)
        turn[role] = content
        self._save()

    def add_tool_call(self, turn_number, tool_called, tool_args, result):
        """
        Record a tool call made during a dialogue turn.

        :param turn_number: the dialogue turn number
        :param tool_called: the name of the tool that was called
        :param tool_args: the arguments passed to the tool
        :param result: the result returned by the tool
        """
        tool_call = {
            "turn": turn_number,
            "tool_called": tool_called,
            "args": tool_args,
            "result": result,
            "timestamp": _now(),
        }

        self.tool_calls.append(tool_call)
        turn = self._get_or_create_turn(turn_number)
        turn["tools"].append(tool_call)

        self.data["tool_calls"].append(tool_call)

        self._save()

    def log_error(self, message):
        """
        :param message: error message to record
        """
        self.data["errors"].append({"timestamp": _now(), "message": message})
        self._save()

    def set_conversation_summary(self, decision, turn):
        """
        Record the final summary for this conversation.

        :param decision: the stopping condition or decision that ended the conversation
        :param turn: the total number of turns taken
        """
        self.data["summary"] = {
            "decision": decision,
            "total_turns": turn,
            "tool_usage": self._tool_usage_summary(),
        }
        self._save()

    def _tool_usage_summary(self):
        counts = defaultdict(int)
        for call in self.tool_calls:
            counts[call["tool_called"]] += 1

        return {
            "total_calls": len(self.tool_calls),
            "by_tool": dict(counts),
            "turns_with_tools": len({c["turn"] for c in self.tool_calls}),
        }

    def _save(self):
        filepath = os.path.join(self.log_dir, f"conversation_{self.conversation_id}.json")
        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)

