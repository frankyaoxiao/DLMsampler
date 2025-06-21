"""
MMaDA Model API for Inspect AI Framework

This module provides a custom ModelAPI implementation for MMaDA that integrates
with the Inspect AI evaluation framework, enabling efficient batch evaluation.
"""

import asyncio
from typing import Any, Union
import sys
import os
import mmada_inference

# Add the parent directory to the path to import mmada_inference
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

from mmada_inference import generate_texts_batch, _load_model, _unload_model


class MMaDAModelAPI(ModelAPI):
    """
    MMaDA Model API for Inspect AI Framework.
    
    This class provides an interface between MMaDA and the Inspect AI framework,
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
        Initialize the MMaDA Model API.
        
        Args:
            model_name: Name of the model (for MMaDA this is typically "mmada-8b-mixcot")
            base_url: Not used for MMaDA (local model)
            api_key: Not used for MMaDA (local model)
            api_key_vars: Not used for MMaDA (local model)
            config: Default generation configuration
            **model_args: Additional model arguments (gen_length, steps, batch_size, etc.)
        """
        super().__init__(model_name, base_url, api_key, api_key_vars, config)
        
        # Store MMaDA-specific configuration
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
        Convert Inspect ChatMessage format to a single prompt string for MMaDA.
        
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
        Generate text using MMaDA.
        
        Args:
            input: List of chat messages
            tools: Tool definitions (not supported by MMaDA)
            tool_choice: Tool choice specification (not supported by MMaDA)
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
            
        # MMaDA doesn't support tools, so we'll just note if they were requested
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
                usage=None  # Token usage tracking not implemented for MMaDA
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
        return f"mmada-{self.model_name}"
    
    def __del__(self):
        """Cleanup when the API instance is destroyed."""
        try:
            # Optionally unload model when API is destroyed
            # Commented out to keep model loaded for efficiency
            # _unload_model()
            pass
        except:
            pass


class MMaDABatchModelAPI(MMaDAModelAPI):
    """
    MMaDA Batch Model API that unloads the model after each batch for memory efficiency.
    
    This is useful when running large evaluations where memory needs to be freed
    between batches.
    """
    
    def __init__(self, *args, unload_after_batch: bool = True, **kwargs):
        """
        Initialize with optional model unloading after batches.
        
        Args:
            unload_after_batch: Whether to unload model after each generation
            *args, **kwargs: Passed to parent MMaDAModelAPI
        """
        super().__init__(*args, **kwargs)
        self._unload_after_batch = unload_after_batch
    
    async def generate(self, input, tools, tool_choice, config) -> ModelOutput:
        """
        Generate with optional model unloading after generation.
        
        Args:
            input: List of chat messages
            tools: Tool definitions (not supported)
            tool_choice: Tool choice specification (not supported)
            config: Generation configuration
            
        Returns:
            ModelOutput with generated text
        """
        # Call parent generate method
        result = await super().generate(input, tools, tool_choice, config)
        
        # Optionally unload model to free memory
        if self._unload_after_batch:
            try:
                await asyncio.get_event_loop().run_in_executor(None, _unload_model)
            except:
                pass  # Ignore cleanup errors
        
        return result 