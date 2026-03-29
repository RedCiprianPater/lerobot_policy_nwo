"""Observation and action processing for NWO policy."""

import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional
import io
import base64


class NWOProcessor:
    """Process observations and actions for NWO API.
    
    Converts LeRobot observation format to NWO-compatible inputs,
    and NWO outputs back to LeRobot action format.
    """
    
    def __init__(
        self,
        instruction_template: str = "Execute robot action based on current observation",
        use_image: bool = True,
    ):
        self.instruction_template = instruction_template
        self.use_image = use_image
    
    def observation_to_instruction(
        self, 
        observation: Dict[str, torch.Tensor],
        task_description: Optional[str] = None
    ) -> str:
        """Convert observation dict to natural language instruction.
        
        Args:
            observation: Dict with keys like 'observation.image', 'observation.state'
            task_description: Optional explicit task description
            
        Returns:
            Natural language instruction for NWO API
        """
        if task_description:
            return task_description
        
        # Extract joint positions if available
        state = observation.get("observation.state")
        if state is not None:
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            # Get most recent state if batched
            if state.ndim > 1:
                state = state[-1]
            joint_str = ", ".join([f"{s:.3f}" for s in state[:6]])
            return f"{self.instruction_template}. Current joint positions: [{joint_str}]"
        
        return self.instruction_template
    
    def observation_to_image_url(
        self,
        observation: Dict[str, torch.Tensor],
    ) -> Optional[str]:
        """Extract and encode image from observation.
        
        Args:
            observation: Dict that may contain 'observation.image'
            
        Returns:
            Base64-encoded image data URL or None
        """
        if not self.use_image:
            return None
        
        image = observation.get("observation.image")
        if image is None:
            return None
        
        # Convert tensor to PIL Image
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Handle batch dimension
        if image.ndim == 4:  # (batch, channels, height, width)
            image = image[-1]  # Take last frame
        
        # Convert from CHW to HWC if needed
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert to PIL and encode
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
    
    def nwo_response_to_action(
        self,
        response: Dict[str, Any],
        expected_shape: Optional[tuple] = None
    ) -> torch.Tensor:
        """Convert NWO API response to LeRobot action tensor.
        
        Args:
            response: JSON response from NWO API
            expected_shape: Expected output shape
            
        Returns:
            Action tensor of shape (n_action_steps, action_dim)
        """
        # Extract actions from NWO response
        actions = response.get("actions", [])
        
        if not actions:
            # Fallback: try to extract from different response format
            action_data = response.get("action", response.get("joint_positions", []))
            if action_data:
                actions = [action_data]
        
        if not actions:
            raise ValueError(f"No actions found in NWO response: {response}")
        
        # Convert to tensor
        action_list = []
        for action in actions:
            if isinstance(action, dict):
                # Extract joint values from dict
                joint_action = [
                    action.get(f"joint_{i}", 0.0) for i in range(6)
                ]
                if "gripper" in action:
                    joint_action.append(action["gripper"])
                action_list.append(joint_action)
            else:
                action_list.append(action)
        
        action_tensor = torch.tensor(action_list, dtype=torch.float32)
        
        # Ensure correct shape (n_action_steps, action_dim)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        
        return action_tensor
    
    def extract_metadata(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract useful metadata from NWO response.
        
        Args:
            response: JSON response from NWO API
            
        Returns:
            Dict with confidence, latency, request_id, etc.
        """
        return {
            "confidence": response.get("confidence", 0.0),
            "request_id": response.get("request_id", ""),
            "instruction": response.get("instruction", ""),
            "latency_ms": response.get("latency_ms", 0),
        }
