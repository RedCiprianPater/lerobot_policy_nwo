# LeRobot Policy for NWO Robotics

[![PyPI version](https://badge.fury.io/py/lerobot_policy_nwo.svg)](https://badge.fury.io/py/lerobot_policy_nwo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Minimal adapter that wraps [NWO Robotics Cloud](https://nwo.capital) API as a [LeRobot](https://github.com/huggingface/lerobot)-compatible policy.

## What This Does

- Converts LeRobot observations → natural language instructions
- Calls NWO's Vision-Language-Action (VLA) API
- Returns actions in LeRobot-compatible format
- Enables LeRobot users to use NWO as a policy backend

## Installation

```bash
pip install lerobot_policy_nwo
```

## Quick Start

### 1. Get API Key

Register at [nwo.capital/webapp/api-key.php](https://nwo.capital/webapp/api-key.php) to get your free API key (100,000 calls/month).

### 2. Basic Usage

```python
from lerobot_policy_nwo import NWOPolicy, NWOPolicyConfig

# Configure policy
config = NWOPolicyConfig(
    api_key="your_nwo_api_key",  # Or set NWO_API_KEY env var
    instruction_template="Pick up the object",
)

# Create policy
policy = NWOPolicy(config)

# Use in LeRobot loop
observation = {
    "observation.image": camera_frame,  # (3, 224, 224) tensor
    "observation.state": joint_positions,  # (6,) tensor
}

action = policy.select_action(observation)
# Returns: tensor of shape (1, action_dim)

# Check API response metadata
metadata = policy.get_last_metadata()
print(f"Confidence: {metadata['confidence']}")
print(f"Request ID: {metadata['request_id']}")
```

### 3. With LeRobot Training

```python
from lerobot.configs.policies import PreTrainedConfig
from lerobot_policy_nwo import NWOPolicy

# Use NWO for data collection
policy = NWOPolicy.from_pretrained("nwo")

# Or specify config
config = PreTrainedConfig.from_pretrained("nwo")
policy = NWOPolicy(config)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `NWO_API_KEY` env | Your NWO API key |
| `api_endpoint` | `https://nwo.capital/webapp` | NWO API base URL |
| `instruction_template` | `"Execute robot action..."` | Template for converting observations |
| `use_image` | `True` | Include camera image in API calls |
| `timeout` | `30.0` | API request timeout (seconds) |

## API Features Used

- **Inference** (`api-robotics.php?action=inference`) - Main VLA inference
- **Health Check** (`?action=health`) - API status
- **Dataset Export** (`?action=export_dataset`) - Export for LeRobot training

## Limitations

- **Inference only**: NWO does not support training. Use LeRobot's imitation learning on collected data.
- **Cloud-based**: Requires internet connection to NWO API
- **Rate limits**: Free tier = 100k calls/month, Prototype = 500k/month

## Development

```bash
git clone https://github.com/nwocapital/lerobot_policy_nwo.git
cd lerobot_policy_nwo
pip install -e ".[dev]"
pytest
```

## Links

- **NWO Robotics**: https://nwo.capital/webapp/nwo-robotics.html
- **API Dashboard**: https://nwo.capital/webapp/api-key.php
- **LeRobot**: https://github.com/huggingface/lerobot
- **PyPI**: https://pypi.org/project/lerobot_policy_nwo/

## License

MIT License - see [LICENSE](LICENSE) file.

## Support

- Email: ciprian.pater@publicae.org
- Discord: NWO Capital community
