"""NWO Robotics Policy for LeRobot.

Minimal adapter that wraps NWO Cloud API as a LeRobot-compatible policy.
Converts observations to instructions, calls NWO inference, returns actions.
"""

from dataclasses import dataclass, field
from typing import Callable

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("nwo")
@dataclass
class NWOPolicyConfig(PreTrainedConfig):
    """Configuration for NWO Robotics Cloud policy.
    
    This policy wraps the NWO Robotics Cloud API, converting LeRobot observations
    to natural language instructions and returning actions from NWO's VLA model.
    
    Args:
        n_obs_steps: Number of observation steps to use as input
        horizon: Action prediction horizon (not used by NWO, always returns 1-step)
        n_action_steps: Number of action steps to execute (always 1 for NWO)
        api_endpoint: NWO API base URL
        api_key: NWO API key for authentication
        instruction_template: Template for converting observations to instructions
        use_image: Whether to include camera image in API call
        timeout: API request timeout in seconds
        normalize: Whether to normalize actions (NWO returns normalized actions)
    """
    # LeRobot standard fields
    n_obs_steps: int = 1
    horizon: int = 1  # NWO returns single actions
    n_action_steps: int = 1
    
    # NWO-specific fields
    api_endpoint: str = "https://nwo.capital/webapp"
    api_key: str = field(default="", repr=False)  # Hide in repr for security
    instruction_template: str = "Execute robot action based on current observation"
    use_image: bool = True
    timeout: float = 30.0
    
    # Feature configuration
    normalize: NormalizationMode = NormalizationMode.IDENTITY
    
    def __post_init__(self):
        super().__post_init__()
        if not self.api_key:
            import os
            self.api_key = os.getenv("NWO_API_KEY", "")
        
        if not self.api_key:
            raise ValueError(
                "NWO API key required. Set api_key in config or NWO_API_KEY environment variable. "
                "Get your key at https://nwo.capital/webapp/api-key.php"
            )
    
    def validate_features(self) -> None:
        """Validate that features are compatible with NWO API."""
        # NWO supports images and joint positions
        if self.image_features is not None:
            for key, feature in self.image_features.items():
                if feature.shape != (3, 224, 224):
                    raise ValueError(
                        f"NWO expects images of shape (3, 224, 224), got {feature.shape} for {key}"
                    )
        
        # NWO returns joint positions or end-effector poses
        if self.action_features is not None:
            for key, feature in self.action_features.items():
                if feature.shape[-1] not in [6, 7, 8]:  # 6D pose, 7D with gripper, 8D with extra
                    raise ValueError(
                        f"NWO action shape should be 6, 7, or 8, got {feature.shape} for {key}"
                    )
