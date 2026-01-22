"""Module matching - Logique de matching entre besoins et prestataires"""

from .matcher import ProviderMatcher
from .text_processor import (
    create_provider_text, 
    create_client_request_text, 
    create_need_text
)

__all__ = [
    'ProviderMatcher',
    'create_provider_text',
    'create_client_request_text',
    'create_need_text'
]
