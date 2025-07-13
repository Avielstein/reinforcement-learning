"""
A3C Competitive Swimmers Agent Module

This module contains the A3C implementation with trust regions for competitive multi-agent learning.
"""

from .networks import (
    ActorCriticNetwork,
    TrustRegionActorCritic,
    SharedGlobalNetwork,
    create_networks
)

from .a3c_agent import (
    A3CWorker,
    A3CManager
)

__all__ = [
    'ActorCriticNetwork',
    'TrustRegionActorCritic', 
    'SharedGlobalNetwork',
    'create_networks',
    'A3CWorker',
    'A3CManager'
]
