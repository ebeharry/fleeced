import asyncio
import litellm

litellm.suppress_debug_info = True
litellm.drop_params = True


def supports_thinking_param(model: str) -> bool:
    """
    Determine whether the given model supports the extended-thinking parameter.

    :param model: model identifier string
    :return: True if the model family supports the thinking param, False otherwise
    """
    model_lower = model.lower()
    return "claude" in model_lower or "qwen" in model_lower


def requires_reasoning_effort_none(model: str) -> bool:
    """
    Determine whether the given model is a gpt-5-family reasoning model that must have
    reasoning_effort explicitly set to "none" to use function tools on the
    /v1/chat/completions endpoint.

    :param model: model identifier string
    :return: True if the model family requires reasoning_effort="none", False otherwise
    """
    model_lower = model.lower()
    return "gpt-5" in model_lower


RETRYABLE_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.BadGatewayError,
    litellm.APIError,
    litellm.BadRequestError,
)


async def async_call_with_retry(
    model: str,
    messages: list,
    max_retries: int = 7,
    base_delay: float = 3.0,
    **kwargs,
) -> litellm.ModelResponse:
    """
    Call litellm.acompletion with exponential backoff on transient errors.

    :param model: Model identifier string.
    :param messages: Message list in OpenAI chat format.
    :param max_retries: Maximum number of attempts before re-raising.
    :param base_delay: Initial sleep duration in seconds; doubles each retry.
    :param kwargs: Additional kwargs forwarded to litellm.acompletion.
    :return: litellm ModelResponse object.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return await litellm.acompletion(model=model, messages=messages, **kwargs)
        except RETRYABLE_EXCEPTIONS:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2
