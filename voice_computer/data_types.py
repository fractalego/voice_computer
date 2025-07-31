"""
Data types for the voice computer system.
"""

from typing import List, Optional, Dict, Any, Callable
from pydantic import BaseModel, Field


class Utterance(BaseModel):
    """Represents a single utterance in a conversation."""
    role: str = Field(
        ...,
        description="The role of the utterance, e.g., 'user', 'assistant', 'system'",
    )
    content: str = Field(..., description="The content of the utterance")


class Messages(BaseModel):
    """Container for conversation messages."""
    utterances: List[Utterance] = Field(
        default_factory=list, description="List of utterances in the conversation"
    )

    def add_system_prompt(self, prompt: str) -> "Messages":
        """Add a system prompt to the beginning of the conversation."""
        system_prompt = Utterance(role="system", content=prompt)
        return Messages(utterances=[system_prompt] + self.utterances)

    def set_system_prompt(self, prompt: str) -> "Messages":
        """Replace or set the system prompt, removing any existing system messages."""
        system_prompt = Utterance(role="system", content=prompt)
        # Filter out any existing system prompts
        non_system_utterances = [u for u in self.utterances if u.role != "system"]
        return Messages(utterances=[system_prompt] + non_system_utterances)

    def add_assistant_utterance(self, content: str) -> "Messages":
        """Add an assistant utterance to the conversation."""
        assistant_utterance = Utterance(role="assistant", content=content)
        return Messages(utterances=self.utterances + [assistant_utterance])

    def add_user_utterance(self, content: str) -> "Messages":
        """Add a user utterance to the conversation."""
        user_utterance = Utterance(role="user", content=content)
        return Messages(utterances=self.utterances + [user_utterance])

    def apply(self, f: Callable) -> "Messages":
        """
        Apply a function to each utterance in the messages.

        Args:
            f: A function that takes an Utterance and returns a modified Utterance.

        Returns:
            A new Messages object with the modified utterances.
        """
        for utterance in self.utterances:
            utterance.content = f(utterance.content)
        return self

    def __repr__(self):
        return "\n".join(
            [f"{utterance.role}: {utterance.content}" for utterance in self.utterances]
        )

    def __str__(self):
        return self.__repr__()


class ToolFunction(BaseModel):
    """Function definition for a tool."""
    name: str = Field(..., description="The name of the function")
    description: str = Field(..., description="A description of what the function does")
    parameters: Dict[str, Any] = Field(
        ..., description="The parameters schema for the function"
    )


class Tool(BaseModel):
    """Tool definition with type and function."""
    type: str = Field(..., description="The type of the tool, e.g., 'function'")
    function: ToolFunction = Field(..., description="The function definition")


class ToolCall(BaseModel):
    """Represents a tool call made by the model."""
    id: str = Field(..., description="Unique identifier for the tool call")
    type: str = Field(..., description="The type of the tool call, e.g., 'function'")
    function: Dict[str, Any] = Field(..., description="The function call details")


class ClientResponse(BaseModel):
    """Response from the client containing messages and tool calls."""
    message: str = Field(..., description="The response message content")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="List of tool calls made by the model"
    )