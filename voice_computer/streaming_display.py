"""
Streaming text display utilities for real-time token output with TTS integration.
"""

import asyncio
import logging

from typing import Optional, Callable
from voice_computer.stopwords import is_stopword

# Utility function to check if a value is numeric
def is_numeric(value):
    """Check if a value is numeric."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

from voice_computer.speaker.tts_speaker import TTSSpeaker

_logger = logging.getLogger(__name__)


class StreamingCompletionException(Exception):
    """Exception to signal completion of streaming display."""
    def __init__(self, message: str = "Streaming display completed"):
        super().__init__(message)
        self.message = message


class StreamingDisplay:
    """Handles streaming text display with configurable batching and output."""
    
    def __init__(
        self,
        batch_size: int = 4,
        flush_delay: float = 1.0,
        output_handler: Optional[Callable[[str], None]] = None,
        end_handler: Optional[Callable[[], None]] = None,
        tts_speaker: Optional[TTSSpeaker] = None,
    ):
        """
        Initialize streaming display.
        
        Args:
            batch_size: Number of tokens to batch before displaying
            flush_delay: Maximum delay before flushing partial batch (seconds)
            output_handler: Custom function to handle text output (default: print)
            end_handler: Custom function to handle end of stream (default: print newline)
        """
        self.batch_size = batch_size
        self.flush_delay = flush_delay
        self.output_handler = output_handler or self._default_output_handler
        self.end_handler = end_handler or self._default_end_handler
        self.tts_speaker = tts_speaker
        self.accumulated_text = ""  # Track all displayed text

    def _default_output_handler(self, text: str) -> None:
        """Default output handler that prints text without newline."""
        self.accumulated_text += text
        print(text, end='', flush=True)
    
    def _default_end_handler(self) -> None:
        """Default end handler that prints a newline."""
        print()
    
    async def display_stream(self, token_queue: asyncio.Queue) -> None:
        """
        Display tokens from a queue in real-time with batching.
        
        Args:
            token_queue: Queue containing tokens to display, None signals end of stream
        """
        token_batch = []
        stream_complete = False
        
        _logger.debug(f"Starting streaming display with batch_size={self.batch_size}, flush_delay={self.flush_delay}")

        # create an async task to handle TTS if a speaker is provided
        is_speaking = False
        speaker_task = None

        while not stream_complete:
            try:
                # Wait for tokens with a timeout to check batching
                token = await asyncio.wait_for(token_queue.get(), timeout=self.flush_delay)
                
                if token is None:  # End of stream signal
                    # Display any remaining tokens
                    if token_batch:
                        remaining_text = ''.join(token_batch)
                        self.output_handler(remaining_text)
                        token_batch.clear()
                    stream_complete = True
                    
                    # Call end handler
                    self.end_handler()
                    
                    break

                # Add token to batch
                token_batch.append(token)
                
                # Display batch if it reaches the batch size
                batch_text = ''.join(token_batch)
                if (len(token_batch) >= self.batch_size and 
                    not is_numeric(token_batch[-1]) and 
                    not is_stopword(token_batch[-1]) and
                    (batch_text[-1].isspace() or batch_text[-1] in '.,!?;:')):
                    self.output_handler(batch_text)
                    token_batch.clear()

                    if not is_speaking and self.tts_speaker is not None:
                        speaker_task = asyncio.create_task(self.tts_speaker.speak_batch())
                        is_speaking = True
                    
                    await asyncio.sleep(0.05)

            except asyncio.TimeoutError:
                # Display partial batch on timeout if there are tokens waiting
                _logger.info("Timeout waiting for streaming display")
                if token_batch:
                    batch_text = ''.join(token_batch)
                    self.output_handler(batch_text)
                    token_batch.clear()
                    
                # Give other async tasks a chance to run
                await asyncio.sleep(0.01)

        # Final safety flush - display any remaining tokens in the batch
        if token_batch:
            remaining_text = ''.join(token_batch)
            self.output_handler(remaining_text)
            token_batch.clear()
            _logger.debug(f"Final flush displayed remaining tokens: '{remaining_text}'")

        if speaker_task is not None:
            _logger.debug(f"Waiting for speaker task to complete. Task done: {speaker_task.done()}")
            try:
                await speaker_task
                _logger.debug("TTS speaker task completed successfully")
            except asyncio.CancelledError:
                # If the speaker task was cancelled, also cancel the TTS playback
                if self.tts_speaker:
                    self.tts_speaker.cancel_playback()
                _logger.debug("TTS speaker task was cancelled")
            except Exception as e:
                _logger.error(f"Error in TTS speaker task: {e}")
        
        _logger.debug("Streaming display completed")
        
        # Signal completion to client by throwing an exception
        raise StreamingCompletionException("Display stream completed")


async def create_streaming_task(
    token_queue: asyncio.Queue,
    batch_size: int = 4,
    flush_delay: float = 0.1,
    output_handler: Optional[Callable[[str], None]] = None,
    end_handler: Optional[Callable[[], None]] = None
) -> asyncio.Task:
    """
    Create and start a streaming display task.
    
    Args:
        token_queue: Queue containing tokens to display
        batch_size: Number of tokens to batch before displaying
        flush_delay: Maximum delay before flushing partial batch (seconds)
        output_handler: Custom function to handle text output
        end_handler: Custom function to handle end of stream
        
    Returns:
        Asyncio task handling the streaming display
    """
    display = StreamingDisplay(
        batch_size=batch_size,
        flush_delay=flush_delay,
        output_handler=output_handler,
        end_handler=end_handler
    )
    
    return asyncio.create_task(display.display_stream(token_queue))


class ColoredStreamingDisplay(StreamingDisplay):
    """Streaming display with colored output support."""
    
    def __init__(
        self,
        batch_size: int = 4,
        flush_delay: float = 1.0,
        color_start: str = "\033[94m",  # Blue
        color_end: str = "\033[0m",     # Reset
        prefix: str = "",
        tts_speaker: Optional[TTSSpeaker] = None,
        **kwargs
    ):
        """
        Initialize colored streaming display.
        
        Args:
            batch_size: Number of tokens to batch before displaying
            flush_delay: Maximum delay before flushing partial batch (seconds)
            color_start: ANSI color code to start with
            color_end: ANSI color code to end with
            prefix: Prefix to add to output (e.g., "bot> ")
            **kwargs: Additional arguments passed to parent class
        """
        self.color_start = color_start
        self.color_end = color_end
        self.prefix = prefix
        self._first_output = True
        self.tts_speaker = tts_speaker

        super().__init__(
            batch_size=batch_size,
            flush_delay=flush_delay,
            tts_speaker=tts_speaker,
            **kwargs
        )
    
    def _default_output_handler(self, text: str) -> None:
        """Colored output handler."""
        # Track clean text for accumulated_text (without colors/prefix)
        clean_text = text
        if self._first_output and self.prefix in text:
            clean_text = text.replace(self.prefix, "")
        clean_text = clean_text.replace(self.color_start, "").replace(self.color_end, "")
        self.accumulated_text += clean_text
        
        if self._first_output:
            output = f"{self.color_start}{self.prefix}{text}"
            self._first_output = False
        else:
            output = text

        if self.tts_speaker:
            tts_text = text.replace(self.color_start, "").replace(self.color_end, "")
            tts_text = tts_text.replace(self.prefix, "")
            self.tts_speaker.add_text_batch(tts_text)

        print(output, end='', flush=True)

    def _default_end_handler(self) -> None:
        """Colored end handler that resets color."""
        print(f"{self.color_end}")
        self._first_output = True  # Reset for next stream


# Convenience functions for common use cases
async def stream_to_console(
    token_queue: asyncio.Queue,
    batch_size: int = 4,
    flush_delay: float = 1.0
) -> tuple[asyncio.Task, StreamingDisplay]:
    """Stream tokens to console with default settings. Returns (task, display_instance)."""
    display = StreamingDisplay(
        batch_size=batch_size,
        flush_delay=flush_delay
    )
    task = asyncio.create_task(display.display_stream(token_queue))
    return task, display


async def stream_colored_to_console(
    token_queue: asyncio.Queue,
    prefix: str = "bot> ",
    color_start: str = "\033[94m",  # Blue
    color_end: str = "\033[0m",     # Reset
    batch_size: int = 4,
    flush_delay: float = 1.0
) -> tuple[asyncio.Task, 'ColoredStreamingDisplay']:
    """Stream tokens to console with colored output. Returns (task, display_instance)."""
    display = ColoredStreamingDisplay(
        batch_size=batch_size,
        flush_delay=flush_delay,
        color_start=color_start,
        color_end=color_end,
        prefix=prefix
    )
    
    task = asyncio.create_task(display.display_stream(token_queue))
    return task, display


async def stream_colored_to_console_with_tts(
        token_queue: asyncio.Queue,
        tts_speaker: TTSSpeaker,
        prefix: str = "bot> ",
        color_start: str = "\033[94m",  # Blue
        color_end: str = "\033[0m",  # Reset
        batch_size: int = 4,
        flush_delay: float = 1.0
) -> tuple[asyncio.Task, 'ColoredStreamingDisplay']:
    """Stream tokens to console with colored output. Returns (task, display_instance)."""
    display = ColoredStreamingDisplay(
        batch_size=batch_size,
        flush_delay=flush_delay,
        color_start=color_start,
        color_end=color_end,
        prefix=prefix,
        tts_speaker=tts_speaker,
    )

    task = asyncio.create_task(display.display_stream(token_queue))
    return task, display

