"""
LLaDA Model API for Inspect AI Framework

This module provides a custom ModelAPI implementation for LLaDA 1.5 that integrates
with the Inspect AI evaluation framework, enabling efficient batch evaluation.
"""

import asyncio
from typing import Any, Union
import sys
import os

# Add the parent directory to the path to import llada_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from inspect_ai.model import (
        ChatMessage, 
        ChatMessageUser, 
        ChatMessageAssistant,
        ChatMessageSystem,
        GenerateConfig, 
        ModelAPI, 
        ModelOutput,
        ChatCompletionChoice
    )
    from inspect_ai.tool import (
        ToolCall,
        ToolChoice,
        ToolInfo
    )
except ImportError as e:
    raise ImportError(
        "Inspect AI is required to use this model API. "
        "Please install it with: pip install inspect-ai"
    ) from e

from llada_inference import generate_texts_batch, _load_model, _unload_model


class LLaDAModelAPI(ModelAPI):
    """
    LLaDA Model API for Inspect AI Framework.
    
    This class provides an interface between LLaDA 1.5 and the Inspect AI framework,
    enabling efficient batch evaluation on datasets like GPQA.
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any
    ) -> None:
        """
        Initialize the LLaDA Model API.
        
        Args:
            model_name: Name of the model (for LLaDA this is typically "llada-1.5")
            base_url: Not used for LLaDA (local model)
            api_key: Not used for LLaDA (local model)
            api_key_vars: Not used for LLaDA (local model)
            config: Default generation configuration
            **model_args: Additional model arguments (gen_length, steps, batch_size, etc.)
        """
        super().__init__(model_name, base_url, api_key, api_key_vars, config)
        
        # Store LLaDA-specific configuration
        self._gen_length = model_args.get("gen_length", 128)
        self._steps = model_args.get("steps", 128) 
        self._block_length = model_args.get("block_length", 32)
        self._temperature = model_args.get("temperature", 0.0)
        self._cfg_scale = model_args.get("cfg_scale", 0.0)
        self._remasking = model_args.get("remasking", "low_confidence")
        self._batch_size = model_args.get("batch_size", 4)
        
        # Load the model once during initialization
        _load_model()
        
    def _format_messages(self, messages: list[ChatMessage]) -> str:
        """
        Convert Inspect ChatMessage format to a single prompt string for LLaDA.
        
        Args:
            messages: List of chat messages from Inspect
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, ChatMessageSystem):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, ChatMessageUser):
                prompt_parts.append(f"User: {message.content}")
            elif isinstance(message, ChatMessageAssistant):
                prompt_parts.append(f"Assistant: {message.content}")
            else:
                # Handle generic ChatMessage
                role = getattr(message, 'role', 'user')
                content = getattr(message, 'content', str(message))
                prompt_parts.append(f"{role.title()}: {content}")
        
        return "\n\n".join(prompt_parts)
    
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """
        Generate text using LLaDA.
        
        Args:
            input: List of chat messages
            tools: Tool definitions (not supported by LLaDA)
            tool_choice: Tool choice specification (not supported by LLaDA)
            config: Generation configuration
            
        Returns:
            ModelOutput with generated text
        """
        # Convert messages to prompt
        prompt = self._format_messages(input)
        
        # Override config with any specified parameters
        gen_length = getattr(config, 'max_tokens', None) or self._gen_length
        temperature = getattr(config, 'temperature', None) 
        if temperature is None:
            temperature = self._temperature
            
        # LLaDA doesn't support tools, so we'll just note if they were requested
        if tools:
            # You could append tool descriptions to the prompt if needed
            # For now, we'll just proceed without tools
            pass
            
        # Generate using our batch function (with batch size 1 for single generation)
        try:
            # Run the generation in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: generate_texts_batch(
                    prompts=[prompt],
                    gen_length=gen_length,
                    steps=self._steps,
                    block_length=self._block_length,
                    temperature=temperature,
                    cfg_scale=self._cfg_scale,
                    remasking=self._remasking,
                    batch_size=1,
                    unload_after=False  # Keep model loaded for efficiency
                )
            )
            
            generated_text = results[0] if results else ""
            
            # Create the assistant message
            assistant_message = ChatMessageAssistant(content=generated_text)
            choice = ChatCompletionChoice(
                message=assistant_message,
                stop_reason="stop"
            )
            
            return ModelOutput(
                choices=[choice],
                usage=None  # Token usage tracking not implemented for LLaDA
            )
            
        except Exception as e:
            # Return an error message if generation fails
            error_message = ChatMessageAssistant(
                content=f"Error during generation: {str(e)}"
            )
            error_choice = ChatCompletionChoice(
                message=error_message,
                stop_reason="error"
            )
            return ModelOutput(
                choices=[error_choice],
                usage=None
            )
    
    def max_connections(self) -> int:
        """Maximum number of concurrent connections (for local model, return 1)."""
        return 1
    
    def connection_key(self) -> str:
        """Key used for connection pooling."""
        return f"llada-{self.model_name}"
    
    def __del__(self):
        """Cleanup when the API instance is destroyed."""
        try:
            # Note: We don't unload the model here by default to allow reuse
            # If you want to unload after each evaluation, you can call _unload_model()
            pass
        except:
            pass


class LLaDABatchModelAPI(LLaDAModelAPI):
    """
    Batch-optimized version of LLaDA Model API for efficient evaluation.
    
    This version is designed for evaluations where you want to process
    multiple samples efficiently and unload the model afterward.
    """
    
    def __init__(self, *args, unload_after_batch: bool = True, **kwargs):
        """
        Initialize batch-optimized LLaDA API.
        
        Args:
            unload_after_batch: Whether to unload model after processing a batch
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self._unload_after_batch = unload_after_batch
        self._batch_counter = 0
        self._batch_threshold = kwargs.get("batch_threshold", 10)  # Unload after N generations
    
    async def generate(self, input, tools, tool_choice, config) -> ModelOutput:
        """Generate with batch tracking."""
        result = await super().generate(input, tools, tool_choice, config)
        
        # Track batch processing
        self._batch_counter += 1
        
        # Unload model after processing threshold batches
        if (self._unload_after_batch and 
            self._batch_counter >= self._batch_threshold):
            try:
                _unload_model()
                self._batch_counter = 0
                # Reload for next batch
                _load_model()
            except:
                pass  # Ignore cleanup errors
                
        return result 