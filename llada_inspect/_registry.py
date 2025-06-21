"""
Registry for LLaDA Inspect AI extensions.

This module imports all the extensions that need to be registered
with Inspect AI through setuptools entry points.
"""

# Import model providers to register them
from .providers import llada, llada_batch

# This file serves as the entry point for setuptools registration
# All imports here will be automatically registered with Inspect AI 