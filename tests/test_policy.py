"""Tests for NWO LeRobot policy adapter."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from lerobot_policy_nwo import NWOPolicy, NWOPolicyConfig, NWOProcessor


class TestNWOPolicyConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test config with environment variable."""
        with patch.dict("os.environ", {"NWO_API_KEY": "test_key_123"}):
            config = NWOPolicyConfig()
            assert config.api_key == "test_key_123"
            assert config.api_endpoint == "https://nwo.capital/webapp"
            assert config.use_image is True
    
    def test_explicit_api_key(self):
        """Test config with explicit API key."""
        config = NWOPolicyConfig(api_key="explicit_key")
        assert config.api_key == "explicit_key"
    
    def test_missing_api_key_raises(self):
        """Test that missing API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="NWO API key required"):
                NWOPolicyConfig()


class TestNWOProcessor:
    """Test observation/action processing."""
    
    @pytest.fixture
    def processor(self):
        return NWOProcessor(
            instruction_template="Execute robot action",
            use_image=True,
        )
    
    def test_observation_to_instruction_with_task(self, processor):
        """Test instruction generation with explicit task."""
        obs = {"observation.state": torch.randn(6)}
        instruction = processor.observation_to_instruction(
            obs, task_description="Pick up the red cube"
        )
        assert instruction == "Pick up the red cube"
    
    def test_observation_to_instruction_without_task(self, processor):
        """Test instruction generation from state."""
        state = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        obs = {"observation.state": state}
        instruction = processor.observation_to_instruction(obs)
        assert "Execute robot action" in instruction
        assert "0.100" in instruction  # Check formatting
    
    def test_observation_to_image_url(self, processor):
        """Test image encoding."""
        # Create fake image tensor (CHW format)
        image = torch.rand(3, 224, 224)
        obs = {"observation.image": image}
        
        url = processor.observation_to_image_url(obs)
        assert url is not None
        assert url.startswith("data:image/jpeg;base64,")
    
    def test_nwo_response_to_action_dict_format(self, processor):
        """Test action extraction from dict response."""
        response = {
            "actions": [
                {"joint_0": 0.1, "joint_1": 0.2, "joint_2": 0.3,
                 "joint_3": 0.4, "joint_4": 0.5, "joint_5": 0.6, "gripper": 0.7}
            ]
        }
        action = processor.nwo_response_to_action(response)
        assert action.shape == (1, 7)
        assert torch.allclose(action[0], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    
    def test_nwo_response_to_action_list_format(self, processor):
        """Test action extraction from list response."""
        response = {
            "actions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        }
        action = processor.nwo_response_to_action(response)
        assert action.shape == (1, 6)
    
    def test_extract_metadata(self, processor):
        """Test metadata extraction."""
        response = {
            "confidence": 0.94,
            "request_id": "req_abc123",
            "instruction": "pick up",
            "latency_ms": 45,
        }
        metadata = processor.extract_metadata(response)
        assert metadata["confidence"] == 0.94
        assert metadata["request_id"] == "req_abc123"


class TestNWOPolicy:
    """Test policy class."""
    
    @pytest.fixture
    def config(self):
        return NWOPolicyConfig(
            api_key="test_key",
            api_endpoint="https://test.nwo.capital/webapp",
        )
    
    @pytest.fixture
    def policy(self, config):
        return NWOPolicy(config)
    
    def test_initialization(self, policy):
        """Test policy initialization."""
        assert policy.config.api_key == "test_key"
        assert policy.processor is not None
        assert policy.last_metadata is None
    
    @patch("requests.Session.post")
    def test_select_action_success(self, mock_post, policy):
        """Test successful action selection."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "actions": [{"joint_0": 0.1, "joint_1": 0.2, "joint_2": 0.3,
                        "joint_3": 0.4, "joint_4": 0.5, "joint_5": 0.6}],
            "confidence": 0.95,
            "request_id": "req_test123",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Create observation
        obs = {
            "observation.state": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        
        # Call select_action
        action = policy.select_action(obs, task_description="Test task")
        
        # Verify
        assert action.shape == (1, 6)
        assert policy.last_metadata is not None
        assert policy.last_metadata["confidence"] == 0.95
    
    @patch("requests.Session.post")
    def test_select_action_api_error(self, mock_post, policy):
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        obs = {"observation.state": torch.randn(6)}
        
        with pytest.raises(RuntimeError, match="NWO API error"):
            policy.select_action(obs)
    
    def test_forward_raises(self, policy):
        """Test that forward raises NotImplementedError."""
        batch = {"observation.state": torch.randn(4, 6)}
        with pytest.raises(NotImplementedError, match="does not support training"):
            policy.forward(batch)
    
    @patch("requests.Session.get")
    def test_health_check(self, mock_get, policy):
        """Test health check."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "version": "1.0"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = policy.health_check()
        assert result["status"] == "ok"
    
    def test_save_pretrained(self, policy, tmp_path):
        """Test saving policy config."""
        save_dir = tmp_path / "saved_policy"
        policy.save_pretrained(save_dir)
        
        # Check files created
        assert (save_dir / "config.json").exists()
        assert (save_dir / "README.md").exists()
        
        # Check API key not saved
        import json
        with open(save_dir / "config.json") as f:
            config = json.load(f)
        assert config["api_key"] == ""


class TestIntegration:
    """Integration-style tests."""
    
    def test_full_workflow_mocked(self):
        """Test full workflow with mocked API."""
        with patch.dict("os.environ", {"NWO_API_KEY": "integration_test_key"}):
            config = NWOPolicyConfig()
            policy = NWOPolicy(config)
            
            # Mock the API call
            with patch.object(policy.session, "post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "actions": [{"joint_0": 0.5, "joint_1": -0.3, "joint_2": 0.2,
                                "joint_3": 0.1, "joint_4": -0.2, "joint_5": 0.4,
                                "gripper": 0.8}],
                    "confidence": 0.92,
                    "request_id": "req_integration_001",
                    "instruction": "Pick up the object",
                }
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response
                
                # Simulate observation from real robot
                observation = {
                    "observation.image": torch.rand(3, 224, 224),
                    "observation.state": torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2, -0.3]),
                }
                
                # Get action
                action = policy.select_action(
                    observation,
                    task_description="Pick up the blue cube"
                )
                
                # Verify
                assert action.shape == (1, 7)  # 6 joints + gripper
                assert policy.get_last_metadata()["confidence"] == 0.92
