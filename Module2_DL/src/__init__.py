"""
Module 2: Baseline Deep Learning (CNN/LSTM) for Toxic Comment Classification
"""

__version__ = "1.0.0"
__author__ = "SE405 Team"

from .preprocess import TextPreprocessor, load_and_split_data, prepare_sequences
from .models import (
    create_embedding_matrix,
    build_cnn_model,
    build_bilstm_model,
    get_callbacks
)
from .data import get_label_distribution, print_label_distribution

__all__ = [
    'TextPreprocessor',
    'load_and_split_data',
    'prepare_sequences',
    'create_embedding_matrix',
    'build_cnn_model',
    'build_bilstm_model',
    'get_callbacks',
    'get_label_distribution',
    'print_label_distribution'
]
