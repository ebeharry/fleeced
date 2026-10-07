"""
Tests for the generation pipeline: DialogueRunner + ConversationLogger integration.
Validates turn structure, tool call limits, stopping conditions, and non-empty text.
"""
import sys
import os
import json
import asyncio
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine.dialogue_runner import DialogueRunner
from src.logging.conversation_logger import ConversationLogger


def make_message(content, tool_calls=None):
    """
    Build a mock LLM message object matching litellm's response structure.
    """
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    return msg


def make_response(content, tool_calls=None):
    """
    Build a mock litellm completion response.
    """
    msg = make_message(content, tool_calls)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def make_tool_call(name, arguments, call_id="tc_1"):
    """
    Build a mock tool call object matching litellm's tool_call structure.
    """
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


class FakeScenario:
    """
    Minimal scenario mock for testing DialogueRunner without real inventory data.
    """

    def __init__(self, stopping_conditions=None, tools=None, opening_messages=None):
        self._stopping_conditions = stopping_conditions or ["<YES>", "<NO>"]
        self._tools = tools
        self._opening_messages = opening_messages or {"agent": "Hello, how may I help you today?"}

    def setup(self, seed):
        pass

    def system_prompts(self):
        return {
            "agent": [{"role": "system", "content": "You are a sales agent."}],
            "user": [{"role": "system", "content": "You are a buyer."}],
        }

    def scenario_parameters(self):
        return {"buyer_preferences": {"budget": 50}, "agent_goal": "sell products"}

    def stopping_conditions(self):
        return self._stopping_conditions

    def opening_messages(self):
        return self._opening_messages

    def tools(self):
        return self._tools

    def tool_handler(self, tool_name, tool_args):
        return {"result": f"mock result for {tool_name}"}


@pytest.fixture
def log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def build_runner(log_dir, scenario=None, max_turns=5, max_tool_calls=3):
    """
    Create a DialogueRunner with a ConversationLogger writing to a temp directory.
    """
    scenario = scenario or FakeScenario()
    logger = ConversationLogger(
        log_dir=log_dir,
        conversation_id="001",
        scenario_type="test_scenario",
        seed=42,
    )
    runner = DialogueRunner(
        model="test-model",
        scenario=scenario,
        logger=logger,
        max_turns=max_turns,
        seed=42,
        max_tool_calls=max_tool_calls,
        verbose=False,
        sleep_time=0,
    )
    return runner, logger


def read_logged_json(log_dir):
    filepath = os.path.join(log_dir, "conversation_001.json")
    with open(filepath) as f:
        return json.load(f)




class TestTurnStructure:
    def test_turn_has_user_and_agent_keys(self, log_dir):
        """
        Each dialogue turn should have 'user', 'agent', and 'tools' keys.
        """
        responses = [
            make_response("I want comfortable shoes"),
            make_response("Here are some options"),
            make_response("I'll take those <YES>"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None))
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        for entry in data["dialogue"]:
            assert "user" in entry
            assert "agent" in entry
            assert "tools" in entry
            assert "turn" in entry

    def test_user_and_agent_share_same_turn_number(self, log_dir):
        """
        Within a conversation turn, user and agent should share the same turn number.
        """
        responses = [
            make_response("I want shoes"),
            make_response("Here are options"),
            make_response("I'll buy them <YES>"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None))
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        non_opening = [t for t in data["dialogue"] if t["turn"] > 0]
        for entry in non_opening:
            if entry["agent"] is not None:
                assert entry["user"] is not None




class TestTurnCount:
    def test_stops_at_stopping_condition(self, log_dir):
        """
        Conversation should stop when user emits a stopping condition token.
        """
        responses = [
            make_response("I want shoes"),
            make_response("Here are options"),
            make_response("Sounds good, I'll take them <YES>"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None), max_turns=10)
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        assert data["summary"]["total_turns"] == 2
        dialogue_turns = [t for t in data["dialogue"] if t["turn"] > 0]
        assert len(dialogue_turns) == 2

    def test_timeout_at_max_turns(self, log_dir):
        """
        Conversation should timeout when max_turns is reached without a stopping condition.
        """
        max_turns = 3
        responses = []
        for _ in range(max_turns):
            responses.append(make_response("Tell me more"))
            responses.append(make_response("Here is more info"))

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None), max_turns=max_turns)
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        assert data["summary"]["decision"] == "Max turns reached without stopping condition"
        assert data["summary"]["total_turns"] == max_turns

    @pytest.mark.parametrize("max_turns", [1, 3, 5])
    def test_turn_numbers_are_sequential(self, log_dir, max_turns):
        """
        Turn numbers in dialogue should be sequential starting from 0 (opening) or 1.
        """
        responses = []
        for i in range(max_turns):
            responses.append(make_response(f"User message {i}"))
            if i == max_turns - 1:
                responses[-1] = make_response(f"Final message <YES>")
            else:
                responses.append(make_response(f"Agent message {i}"))

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None), max_turns=max_turns)
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        turn_numbers = [t["turn"] for t in data["dialogue"]]
        for i in range(1, len(turn_numbers)):
            assert turn_numbers[i] >= turn_numbers[i - 1]




class TestMaxToolCalls:
    def test_tool_calls_do_not_exceed_max(self, log_dir):
        """
        No single turn should have more tool calls than max_tool_calls.
        """
        max_tool_calls = 3
        fake_tools = [{"type": "function", "function": {"name": "search_inventory"}}]

        tool_calls = [
            make_tool_call("search_inventory", {"query": "shoes"}, f"tc_{i}")
            for i in range(5)
        ]
        responses = [
            make_response("I need shoes"),
            make_response(None, tool_calls=tool_calls),
            make_response("Here are the results"),
            make_response("I'll take them <YES>"),
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            scenario = FakeScenario(tools=fake_tools)
            runner, _ = build_runner(log_dir, scenario=scenario, max_tool_calls=max_tool_calls)
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        for entry in data["dialogue"]:
            assert len(entry["tools"]) <= max_tool_calls

    @pytest.mark.parametrize("max_tool_calls", [1, 2, 3])
    def test_respects_different_max_tool_call_limits(self, log_dir, max_tool_calls):
        """
        Tool call limit should be enforced regardless of the configured max.
        """
        fake_tools = [{"type": "function", "function": {"name": "search_inventory"}}]

        tool_calls = [
            make_tool_call("search_inventory", {"query": "shoes"}, f"tc_{i}")
            for i in range(5)
        ]
        responses = [
            make_response("I need shoes"),
            make_response(None, tool_calls=tool_calls),
            make_response("Here are the results"),
            make_response("I'll take them <YES>"),
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            scenario = FakeScenario(tools=fake_tools)
            runner, _ = build_runner(log_dir, scenario=scenario, max_tool_calls=max_tool_calls)
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        for entry in data["dialogue"]:
            assert len(entry["tools"]) <= max_tool_calls

    def test_tool_calls_logged_on_correct_turn(self, log_dir):
        """
        Tool calls should be logged on the same turn as the agent response that used them.
        """
        fake_tools = [{"type": "function", "function": {"name": "get_product_details"}}]
        tool_calls = [make_tool_call("get_product_details", {"product_id": "B123"}, "tc_1")]

        responses = [
            make_response("Show me shoes"),
            make_response(None, tool_calls=tool_calls),
            make_response("Here are the details"),
            make_response("I'll buy them <YES>"),
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            scenario = FakeScenario(tools=fake_tools)
            runner, _ = build_runner(log_dir, scenario=scenario)
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        turns_with_tools = [t for t in data["dialogue"] if t["tools"]]
        for entry in turns_with_tools:
            assert entry["agent"] is not None
            for tc in entry["tools"]:
                assert tc["turn"] == entry["turn"]




class TestStoppingCondition:
    def test_summary_has_stopping_condition(self, log_dir):
        """
        Summary decision should match the stopping condition token found in user text.
        """
        responses = [
            make_response("I want shoes"),
            make_response("Here are options"),
            make_response("No thanks <NO>"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None))
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        assert data["summary"] is not None
        assert data["summary"]["decision"] == "<NO>"

    def test_yes_stopping_condition(self, log_dir):
        """
        <YES> token in user text should end the conversation with decision '<YES>'.
        """
        responses = [
            make_response("Show me shoes"),
            make_response("Here you go"),
            make_response("I'll take them <YES>"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None))
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        assert data["summary"]["decision"] == "<YES>"

    def test_timeout_when_no_stopping_condition(self, log_dir):
        """
        Without a stopping condition token, conversation should timeout.
        """
        responses = []
        for _ in range(2):
            responses.append(make_response("Tell me more"))
            responses.append(make_response("Here is info"))

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None), max_turns=2)
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        assert data["summary"]["decision"] == "Max turns reached without stopping condition"




class TestNoEmptyText:
    def test_user_text_is_not_empty(self, log_dir):
        """
        User text in each turn should not be empty or None (except opening-only turns).
        """
        responses = [
            make_response("I want shoes please"),
            make_response("Here are some options"),
            make_response("Great, I'll take them <YES>"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None))
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        for entry in data["dialogue"]:
            if entry["turn"] > 0 and entry["user"] is not None:
                assert entry["user"].strip() != ""

    def test_agent_text_is_not_empty(self, log_dir):
        """
        Agent text in turns where agent responds should not be empty.
        """
        responses = [
            make_response("I want shoes"),
            make_response("Here are some comfortable options for you"),
            make_response("I'll buy those <YES>"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None))
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        for entry in data["dialogue"]:
            if entry["agent"] is not None:
                assert entry["agent"].strip() != ""

    def test_opening_message_is_not_empty(self, log_dir):
        """
        Opening messages should not be empty.
        """
        responses = [
            make_response("Hi, I need shoes"),
            make_response("Let me help you"),
            make_response("Sounds good <YES>"),
        ]
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=responses), \
             patch("time.sleep"):
            runner, _ = build_runner(log_dir, scenario=FakeScenario(tools=None))
            asyncio.run(runner.run())

        data = read_logged_json(log_dir)
        opening = [t for t in data["dialogue"] if t["turn"] == 0]
        for entry in opening:
            if entry["agent"] is not None:
                assert entry["agent"].strip() != ""
            if entry["user"] is not None:
                assert entry["user"].strip() != ""
