import json
import numpy as np
import asyncio
from src.logging.conversation_logger import ConversationLogger
from src.utils.litellm_utils import (
    RETRYABLE_EXCEPTIONS,
    async_call_with_retry,
    requires_reasoning_effort_none,
    supports_thinking_param,
)

_DEEPSEEK_SENTINEL = "<｜｜DSML｜｜"


def _strip_sentinel(text: str) -> str:
    """
    Strip a message's text if it contains the deepseek special token sentinel.

    :param text: raw text produced by the model
    :return: text unchanged, or an empty string if the sentinel is present
    """
    return "" if _DEEPSEEK_SENTINEL in text else text


class DialogueRunner:
    """
    A scenario-agnostic dialogue engine to manage each dialogue turn, tool use, logging, and stopping conditions.
    """

    def __init__(self,
                 model: str, 
                 scenario, 
                 logger,
                 max_turns: int,
                 seed: int,
                 max_tool_calls: int = 3,
                 verbose: bool = False,
                 sleep_time: float = 1.0):
        """
        :param model: the string name of the model to be used for both agent and user LLM
        :param scenario: the scenario being run by the dialogue runner
        :param logger: the logger used for the dialogue runner
        :param max_turns: max number of total turns a dialogue can have
        :param verbose: whether or not model is verbose
        :param sleep_time: sleep time, in seconds, after each model or tool call, used to throttle API request rate
        """
        self.model = model 
        self.logger = logger

        self.scenario = scenario
        self.stopping_conditions = self.scenario.stopping_conditions()
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls

        self.sleep_time = sleep_time
        self.seed = seed
        self.verbose = verbose

    async def _call_model(self, role_dict: dict, role: str, tools: list | None, tool_choice: str | None):
        """
        Complete one model call and log any errors. Retries on transient failures with exponential backoff.
        """
        try:
            return await async_call_with_retry(
                model=self.model,
                messages=role_dict[role],
                tools=tools,
                tool_choice=tool_choice,
                temperature=1,
                extra_body={"enable_thinking": False} if supports_thinking_param(self.model) else {},
                reasoning_effort="none" if requires_reasoning_effort_none(self.model) else None,
            )
        except RETRYABLE_EXCEPTIONS as e:
            self.logger.log_error(f"API call failed. Max retries reached with {e}")
            raise
        except Exception as e:
            self.logger.log_error(f"API call failed with error {e}")
            raise
    
    async def _get_model_message(self, role_dict: dict, role: str, tools: list | None, tool_choice: str | None):
        """
        Calls model and checks if response is empty.
        """
        response = await self._call_model(role_dict, role, tools, tool_choice)
        await asyncio.sleep(self.sleep_time) 

        if not response.choices:
            raise ValueError("API returned empty choices")
        message = response.choices[0].message
        return message

    def _record_message(self, role_dict: dict, role: str, text: str, own_tag: str, reasoning_content: str | None = None):
        """
        Append a message to the role's own conversation history and, if non-empty, to the
        opposite role's conversation history as a "user" turn.

        :param role_dict: a dictionary with the keys assistant and user, and the values being the conversation log lists
        :param role: the role whose history the message was generated for, "agent" or "user"
        :param text: the message text to record
        :param own_tag: the "role" field to use when appending to role_dict[role]
        :param reasoning_content: thinking-mode reasoning content to round-trip back to the API
            (e.g. required by DeepSeek reasoner models), omitted from the opposite role's history
        """
        other_key = "agent" if role == "user" else "user"
        own_message = {"role": own_tag, "content": text}
        if reasoning_content:
            own_message["reasoning_content"] = reasoning_content
        role_dict[role].append(own_message)
        if text:
            role_dict[other_key].append({"role": "user", "content": text})

    async def _prompt_and_log(self, role_dict: dict, role: str, turn_number: int, tools=None):
        """
        Prompt model and save its message to the conversation log for both models.
        Handles tool calls for the agent role and logs them.

        :param role_dict: a dictionary with the keys assistant and user, and the values being the conversation log lists
        :param role: the role of the model, "assistant" when generating a response, and "user" when receiving input
        :param turn_number: the number of the turn for the logger
        :param tools: self.scenario tools if available
        """
        use_tools = bool(tools)
        tool_choice = "auto" if tools else None

        message = await self._get_model_message(role_dict, role, tools, tool_choice)

        tool_call_count = 0
        while use_tools and getattr(message, "tool_calls", None):

            if tool_call_count >= self.max_tool_calls:
                if self.verbose:
                    print(f"Max tool calls ({self.max_tool_calls}) reached, forcing final response.")
                message = await self._get_model_message(role_dict, role, None, None)
                break

            remaining = self.max_tool_calls - tool_call_count
            tool_calls_to_process = message.tool_calls[:remaining]

            tool_call_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_to_process,
            }
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content:
                tool_call_message["reasoning_content"] = reasoning_content
            role_dict[role].append(tool_call_message)

            for tool_call in tool_calls_to_process:
                tool_call_count += 1
                tool_name = tool_call.function.name

                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    self.logger.log_error(f"Tool argument parsing failed for {tool_name}: {e}")
                    result = {"error": f"Invalid tool arguments: {e}"}
                    result_json = json.dumps(result)
                    role_dict[role].append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result_json,
                        }
                    )
                    continue

                if self.verbose:
                    print(f" --> Tool call ({tool_call_count}/{self.max_tool_calls}): {tool_name}({tool_args})")

                result = None
                try:
                    result = self.scenario.tool_handler(tool_name, tool_args)
                    result_json = json.dumps(result)
                except (TypeError, ValueError) as e:
                    self.logger.log_error(f"Tool result serialization failed: {e}")
                    result = {"error": str(e)}
                    result_json = json.dumps(result)

                await asyncio.sleep(self.sleep_time)
                self.logger.add_tool_call(turn_number, tool_name, tool_args, result)

                role_dict[role].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result_json,
                    }
                )

            message = await self._get_model_message(role_dict, role, tools, tool_choice)

        text = _strip_sentinel(message.content or "")
        reasoning_content = getattr(message, "reasoning_content", None)

        if not text and use_tools:
            try:
                retry_message = await self._get_model_message(role_dict, role, None, None)
                text = _strip_sentinel(retry_message.content or "")
                reasoning_content = getattr(retry_message, "reasoning_content", None)
            except Exception:
                pass

        if not text and role == "agent":
            text = "I'm sorry, I wasn't able to process that. Could you repeat your question?"

        if self.verbose and text:
            print(f"**{role}**: {text}\n")

        self._record_message(role_dict, role, text, "assistant", reasoning_content=reasoning_content)

        self.logger.add_turn(turn_number, role, text)

        if role == "user" and self.stopping_conditions:
            for cond in self.stopping_conditions:
                if cond in text:
                    return role_dict, False, cond

        return role_dict, True, None
    
    async def run(self):
        """
        Initializes scenario and dialogue state, and runs the two-agent conversation between "agent" LLM and "user" LLM
        The role variable of an agent refers to whether it is the agent or the user in the scenario.
        The role in the role_dict dictionary content can be assistant or user, and refers to whether the text
        was generated by the model, or received by the model. 
        """
        self.scenario.setup(self.seed)
        tools = self.scenario.tools()
        self.logger.setup(self.scenario.scenario_parameters())

        role_dict = self.scenario.system_prompts()
        openings = self.scenario.opening_messages()
        
        if self.verbose:
            print("=== CONVERSATION START ===\n")

        if openings:
            for role, text in openings.items():
                own_tag = "assistant" if role == "agent" else "user"
                self._record_message(role_dict, role, text, own_tag)

                self.logger.add_turn(0, role, text)

                if self.verbose:
                    print(f"**{role}**: {text}\n")
        
        for turn in range(1, self.max_turns + 1):
            if self.verbose:
                print(f"--- Turn {turn} ---")

            role_dict, cont, stopping_condition = await self._prompt_and_log(
                role="user",
                role_dict=role_dict,
                turn_number=turn,
            )

            if not cont:
                if self.verbose:
                    print("\n=== CONVERSATION END ===\n")
                self.logger.set_conversation_summary(stopping_condition, turn)
                return

            role_dict, cont, stopping_condition = await self._prompt_and_log(
                role="agent",
                role_dict=role_dict,
                turn_number=turn,
                tools=tools
            )

        self.logger.set_conversation_summary("Max turns reached without stopping condition", turn)
        if self.verbose:
                print("=== CONVERSATION END ===\n")
