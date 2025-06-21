"""
LLaDA extensions for Inspect AI.

This package provides custom model APIs for using LLaDA 1.5 with the Inspect AI
evaluation framework, enabling efficient evaluation on datasets like GPQA.
"""

from .llada_api import LLaDAModelAPI, LLaDABatchModelAPI
from .providers import llada, llada_batch

__all__ = [
    "LLaDAModelAPI", 
    "LLaDABatchModelAPI",
    "llada", 
    "llada_batch"
] 