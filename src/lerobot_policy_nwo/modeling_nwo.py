"""NWO Policy model for LeRobot.

Implements PreTrainedPolicy interface to integrate NWO Robotics Cloud
with LeRobot training and evaluation pipelines.
"""

import json
import requests
from typing import Dict, Any, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
from lerobot.common.policies.pretrained_policy import PreTrainedPolicy

from .configuration_nwo import NWOPolicyConfig
from .processor_nwo import NWOProcessor


class NWOPolicy(PreTrainedPolicy):
    """NWO Robotics Cloud policy adapter for LeRobot.
    
    This policy wraps the NWO Robotics Cloud API, allowing LeRobot users to:
    - Use NWO's VLA model for inference
    - Collect data with NWO-powered robots
    - Evaluate NWO against other policies
    
    The policy converts LeRobot observations to NWO API calls and returns
    actions in LeRobot-compatible format.
    
    Example:
        >>> from lerobot_policy_nwo import NWOPolicy, NWOPolicyConfig
        >>> config = NWOPolicyConfig(api_key="your_key")
        >>> policy = NWOPolicy(config)
        >>> action = policy.select_action(observation)
    """
    
    name = "nwo"
    config_class = NWOPolicyConfig
    
    def __init__(
        self,
        config: NWOPolicyConfig,
        dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ):
        """Initialize NWO policy.
        
        Args:
            config: NWOPolicyConfig with API credentials
            dataset_stats: Optional dataset statistics for normalization
        """
        super().__init__(config, dataset_stats)
        self.config = config
        self.processor = NWOProcessor(
            instruction_template=config.instruction_template,
            use_image=config.use_image,
        )
        
        # Validate API key
        if not self.config.api_key:
            raise ValueError(
                "NWO API key required. Get yours at https://nwo.capital/webapp/api-key.php"
            )
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": self.config.api_key,
            "Content-Type": "application/json",
        })
        
        # API endpoint
        self.inference_url = f"{self.config.api_endpoint}/api-robotics.php"
        
        # Metadata tracking
        self.last_metadata: Optional[Dict[str, Any]] = None
    
    def select_action(
        self,
        observation: Dict[str, torch.Tensor],
        task_description: Optional[str] = None,
    ) -> torch.Tensor:
        """Select action using NWO Cloud API.
        
        Args:
            observation: Dict with observation tensors (image, state, etc.)
            task_description: Optional explicit task instruction
            
        Returns:
            Action tensor of shape (n_action_steps, action_dim)
        """
        # Convert observation to NWO format
        instruction = self.processor.observation_to_instruction(
            observation, task_description
        )
        image_url = self.processor.observation_to_image_url(observation)
        
        # Build API request
        payload = {
            "instruction": instruction,
        }
        if image_url and self.config.use_image:
            payload["image_url"] = image_url
        
        # Make API call
        try:
            response = self.session.post(
                f"{self.inference_url}?action=inference",
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"NWO API request failed: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from NWO API: {e}")
        
        # Check for API errors
        if "error" in data:
            raise RuntimeError(f"NWO API error: {data['error']}")
        
        # Convert response to action tensor
        action = self.processor.nwo_response_to_action(
            data,
            expected_shape=(self.config.n_action_steps, self.config.action_dim)
        )
        
        # Store metadata for inspection
        self.last_metadata = self.processor.extract_metadata(data)
        
        return action
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for training (not supported by NWO).
        
        NWO is an inference-only API. For training, use LeRobot's
        imitation learning on collected data.
        
        Args:
            batch: Batch of observations
            
        Returns:
            Dict with predicted actions
            
        Raises:
            NotImplementedError: NWO does not support training
        """
        raise NotImplementedError(
            "NWO policy does not support training. "
            "Use select_action() for inference or collect data for LeRobot training."
        )
    
    def save_pretrained(
        self,
        save_directory: str | Path,
        **kwargs,
    ) -> None:
        """Save policy configuration.
        
        Note: NWO policy has no weights to save (cloud-based).
        Only saves config for reproducibility.
        
        Args:
            save_directory: Directory to save config
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config (without API key for security)
        config_dict = self.config.to_dict()
        config_dict["api_key"] = ""  # Remove key
        
        config_path = save_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save README
        readme_path = save_path / "README.md"
        readme_path.write_text(
            "# NWO Policy\\n\\n"
            "This policy uses NWO Robotics Cloud API.\\n\\n"
            "To use:\\n"
            "1. Get API key: https://nwo.capital/webapp/api-key.php\\n"
            "2. Set NWO_API_KEY environment variable or pass api_key to config\\n"
        )
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        config: Optional[NWOPolicyConfig] = None,
        dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> "NWOPolicy":
        """Load policy from saved config.
        
        Args:
            pretrained_name_or_path: Path to saved config or "nwo" for default
            config: Optional config to override saved values
            dataset_stats: Dataset statistics
            
        Returns:
            Loaded NWOPolicy
        """
        if str(pretrained_name_or_path).lower() == "nwo":
            # Use default config
            config = config or NWOPolicyConfig()
            return cls(config, dataset_stats)
        
        # Load from path
        config_path = Path(pretrained_name_or_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            
            # Merge with provided config
            if config is None:
                config = NWOPolicyConfig(**config_dict)
        else:
            config = config or NWOPolicyConfig()
        
        return cls(config, dataset_stats)
    
    def get_last_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata from last API call.
        
        Returns:
            Dict with confidence, latency, request_id, etc.
        """
        return self.last_metadata
    
    def health_check(self) -> Dict[str, Any]:
        """Check NWO API health.
        
        Returns:
            Health status dict
        """
        try:
            response = self.session.get(
                f"{self.inference_url}?action=health",
                timeout=5.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
