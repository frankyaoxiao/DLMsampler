"""
Model Providers for LLaDA and MMaDA Models in Inspect AI Framework

This module registers LLaDA and MMaDA models as available providers for the Inspect AI framework.
"""

from inspect_ai.model import modelapi
from .llada_api import LLaDAModelAPI, LLaDABatchModelAPI
from .mmada_api import MMaDAModelAPI, MMaDABatchModelAPI


# Register LLaDA models
@modelapi(name="llada")
def llada():
    """Standard LLaDA Model API for Inspect AI"""
    return LLaDAModelAPI

@modelapi(name="llada-batch") 
def llada_batch_modelapi():
    """LLaDA Batch Model API with memory management for Inspect AI"""
    return LLaDABatchModelAPI

@modelapi(name="llada/llada-1.5")
def llada_1_5_modelapi():
    """LLaDA 1.5 Model API for Inspect AI"""
    return LLaDAModelAPI

@modelapi(name="llada/llada-1.5-batch")
def llada_1_5_batch_modelapi():
    """LLaDA 1.5 Batch Model API with memory management for Inspect AI"""
    return LLaDABatchModelAPI


# Register MMaDA models
@modelapi(name="mmada")
def mmada_modelapi():
    """Standard MMaDA Model API for Inspect AI"""
    return MMaDAModelAPI

@modelapi(name="mmada-batch")
def mmada_batch_modelapi():
    """MMaDA Batch Model API with memory management for Inspect AI"""
    return MMaDABatchModelAPI

@modelapi(name="mmada/mmada-8b-mixcot")
def mmada_8b_mixcot_modelapi():
    """MMaDA 8B MixCoT Model API for Inspect AI"""
    return MMaDAModelAPI

@modelapi(name="mmada/mmada-8b-mixcot-batch")
def mmada_8b_mixcot_batch_modelapi():
    """MMaDA 8B MixCoT Batch Model API with memory management for Inspect AI"""
    return MMaDABatchModelAPI 