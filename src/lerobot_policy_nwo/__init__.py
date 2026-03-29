"""LeRobot Policy Adapter for NWO Robotics Cloud.

This package provides a minimal adapter that wraps the NWO Robotics Cloud API
as a LeRobot-compatible policy. It converts observations to instructions,
calls NWO inference, and returns actions.

Example:
    >>> from lerobot_policy_nwo import NWOPolicy, NWOPolicyConfig
    >>> 
    >>> # Configure with your API key
    >>> config = NWOPolicyConfig(
    ...     api_key="your_nwo_api_key",
    ...     instruction_template="Pick up the object",
    ... )
    >>> 
    >>> # Create policy
    >>> policy = NWOPolicy(config)
    >>> 
    >>> # Use for inference
    >>> action = policy.select_action(observation)
    >>> 
    >>> # Check metadata
    >>> print(policy.get_last_metadata())
    {'confidence': 0.94, 'request_id': 'req_abc123', ...}

Installation:
    pip install lerobot_policy_nwo

Get API Key:
    Visit https://nwo.capital/webapp/api-key.php to register and get your API key.
"""

__version__ = "0.1.0"
__author__ = "NWO Capital"
__email__ = "ciprian.pater@publicae.org"

from .configuration_nwo import NWOPolicyConfig
from .modeling_nwo import NWOPolicy
from .processor_nwo import NWOProcessor

__all__ = [
    "NWOPolicy",
    "NWOPolicyConfig", 
    "NWOProcessor",
]
