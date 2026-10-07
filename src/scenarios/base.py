from abc import ABC, abstractmethod


class BaseScenario(ABC):
    """
    This provides the base template for the two-LLM dialogue between an "agent" LLM,
    whose propensity to be deceptive will be evaluated, and a "user" LLM, who acts as the human
    in each scenario.
    """

    @abstractmethod
    def setup(self, seed: int | None = None) -> None:
        """
        Initializes the specific scenario attributes

        :param seed: randomness seed for the dialogue
        :type seed: int | None
        """

    @abstractmethod
    def system_prompts(self) -> dict[str, str]:
        """
        Returns a dictionary of system prompts keyed by the role name, agent or user.
        The system prompts must be inside of lists.

        :return: Dict mapping role name ("agent" or "user") to its system prompt.
        """

    @abstractmethod
    def stopping_conditions(self) -> list[str]:
        """
        Returns a list of the tokens that signal the end of the dialogue, e.g. <PURCHASE> or <STOP>

        :return: List of stopping condition token strings.
        """

    @abstractmethod
    def scenario_parameters(self) -> dict:
        """
        Returns a dictionary of the parameters for the scenario for logging.

        :return: Dict of scenario parameters to log.
        """

    def opening_messages(self) -> dict[str, str]:
        """
        Returns a dictionary of opening messages keyed by each role. These are dialogue turns and logged.
        The default is to not have an opening.

        e.g. {"user" : "Hello, how may I help you?"}

        If only one role has a opening message, it will default to being the role that speaks first / has the first dialogue turn.

        :return: Dict mapping role name to its opening message; empty by default.
        """
        return {}

    def tools(self) -> list[dict] | None:
        """
        Optional definition of tools for the scenario

        :return: List of tool schema dicts, or None if the scenario has no
            tools.
        """
        return None

    def tool_handler(self, tool_name: str, tool_args: dict) -> dict | None:
        """
        Optional handler if the scenario has tools

        :param tool_name: name of the tool to execute
        :param tool_args: arguments to pass to the tool
        :return: Tool execution result dict, or None if the scenario has no
            tools.
        """
        return None