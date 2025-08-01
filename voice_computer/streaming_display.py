"""
Streaming text display utilities for real-time token output with TTS integration.
"""

import asyncio
import logging
from typing import Optional, Callable, Any

_logger = logging.getLogger(__name__)


class StreamingDisplay:
    """Handles streaming text display with configurable batching and output."""
    
    def __init__(
        self,
        batch_size: int = 4,
        flush_delay: float = 0.1,
        output_handler: Optional[Callable[[str], None]] = None,
        end_handler: Optional[Callable[[], None]] = None
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
        
    def _default_output_handler(self, text: str) -> None:
        """Default output handler that prints text without newline."""
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
        
        while not stream_complete:
            try:
                # Wait for tokens with a timeout to check batching
                token = await asyncio.wait_for(token_queue.get(), timeout=self.flush_delay)
                
                if token is None:  # End of stream signal
                    stream_complete = True
                    # Display any remaining tokens
                    if token_batch:
                        remaining_text = ''.join(token_batch)
                        self.output_handler(remaining_text)
                        token_batch.clear()
                    
                    # Call end handler
                    self.end_handler()
                    break

                # Add token to batch
                token_batch.append(token)
                
                # Display batch if it reaches the batch size
                if len(token_batch) >= self.batch_size:
                    batch_text = ''.join(token_batch)
                    self.output_handler(batch_text)
                    token_batch.clear()
                    
            except asyncio.TimeoutError:
                # Display partial batch on timeout if there are tokens waiting
                if token_batch:
                    batch_text = ''.join(token_batch)
                    self.output_handler(batch_text)
                    token_batch.clear()
        
        _logger.debug("Streaming display completed")


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
        flush_delay: float = 0.1,
        color_start: str = "\033[94m",  # Blue
        color_end: str = "\033[0m",     # Reset
        prefix: str = "",
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
        
        super().__init__(
            batch_size=batch_size,
            flush_delay=flush_delay,
            **kwargs
        )
    
    def _default_output_handler(self, text: str) -> None:
        """Colored output handler."""
        if self._first_output:
            # Add prefix and color start for first output
            output = f"{self.color_start}{self.prefix}{text}"
            self._first_output = False
        else:
            output = text
        
        print(output, end='', flush=True)
    
    def _default_end_handler(self) -> None:
        """Colored end handler that resets color."""
        print(f"{self.color_end}")
        self._first_output = True  # Reset for next stream


# Convenience functions for common use cases
async def stream_to_console(
    token_queue: asyncio.Queue,
    batch_size: int = 4,
    flush_delay: float = 0.1
) -> asyncio.Task:
    """Stream tokens to console with default settings."""
    return await create_streaming_task(
        token_queue=token_queue,
        batch_size=batch_size,
        flush_delay=flush_delay
    )


async def stream_colored_to_console(
    token_queue: asyncio.Queue,
    prefix: str = "bot> ",
    color_start: str = "\033[94m",  # Blue
    color_end: str = "\033[0m",     # Reset
    batch_size: int = 4,
    flush_delay: float = 0.1
) -> asyncio.Task:
    """Stream tokens to console with colored output."""
    display = ColoredStreamingDisplay(
        batch_size=batch_size,
        flush_delay=flush_delay,
        color_start=color_start,
        color_end=color_end,
        prefix=prefix
    )
    
    return asyncio.create_task(display.display_stream(token_queue))


class TTSStreamingDisplay(ColoredStreamingDisplay):
    """Streaming display with TTS integration for real-time speech synthesis."""
    
    def __init__(
        self,
        tts_speaker,
        batch_size: int = 4,
        flush_delay: float = 0.1,
        color_start: str = "\033[94m",  # Blue
        color_end: str = "\033[0m",     # Reset
        prefix: str = "",
        **kwargs
    ):
        """
        Initialize TTS streaming display.
        
        Args:
            tts_speaker: TTSSpeaker instance for speech synthesis
            batch_size: Number of tokens to batch before displaying
            flush_delay: Maximum delay before flushing partial batch (seconds)
            color_start: ANSI color code to start with
            color_end: ANSI color code to end with
            prefix: Prefix to add to output (e.g., "bot> ")
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            batch_size=batch_size,
            flush_delay=flush_delay,
            color_start=color_start,
            color_end=color_end,
            prefix=prefix,
            **kwargs
        )
        self.tts_speaker = tts_speaker
        self._tts_active = False
    
    async def display_stream(self, token_queue: asyncio.Queue) -> None:
        """
        Display tokens from a queue in real-time with TTS synthesis.
        
        Args:
            token_queue: Queue containing tokens to display, None signals end of stream
        """
        # Start TTS streaming
        if self.tts_speaker:
            try:
                self.tts_speaker.start_streaming_speech(text_callback=self._tts_text_callback)
                self._tts_active = True
                _logger.debug("TTS streaming started")
            except Exception as e:
                _logger.error(f"Failed to start TTS streaming: {e}")
                self.tts_speaker = None
        
        try:
            # Use parent's display logic but with TTS integration
            await super().display_stream(token_queue)
            await asyncio.sleep(0.1)
        finally:
            # Stop TTS streaming
            if self._tts_active and self.tts_speaker:
                try:
                    self.tts_speaker.stop_streaming_speech()
                    self._tts_active = False
                    _logger.debug("TTS streaming stopped")
                except Exception as e:
                    _logger.error(f"Error stopping TTS streaming: {e}")
    
    def _default_output_handler(self, text: str) -> None:
        """Output handler with TTS integration."""
        # Send text to TTS for synthesis
        if self._tts_active and self.tts_speaker and text.strip():
            try:
                self.tts_speaker.add_text_batch(text)
            except Exception as e:
                _logger.error(f"Error adding text to TTS: {e}")
        
        # Display text normally
        super()._default_output_handler(text)
    
    def _tts_text_callback(self, text_batch: str):
        """Callback for TTS text processing (for debugging/logging)."""
        pass


# Convenience function for TTS streaming
async def stream_with_tts_to_console(
    token_queue: asyncio.Queue,
    tts_speaker,
    prefix: str = "bot> ",
    color_start: str = "\033[94m",  # Blue
    color_end: str = "\033[0m",     # Reset
    batch_size: int = 4,
    flush_delay: float = 0.1
) -> asyncio.Task:
    """Stream tokens to console with TTS synthesis."""
    display = TTSStreamingDisplay(
        tts_speaker=tts_speaker,
        batch_size=batch_size,
        flush_delay=flush_delay,
        color_start=color_start,
        color_end=color_end,
        prefix=prefix
    )
    
    return asyncio.create_task(display.display_stream(token_queue))