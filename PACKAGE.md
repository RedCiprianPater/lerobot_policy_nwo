# lerobot_policy_nwo

Minimal LeRobot policy adapter for NWO Robotics Cloud API.

## Package Structure

```
lerobot_policy_nwo/
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI testing
│       └── publish.yml      # PyPI publishing
├── src/
│   └── lerobot_policy_nwo/
│       ├── __init__.py              # Package exports
│       ├── configuration_nwo.py     # NWOPolicyConfig
│       ├── modeling_nwo.py          # NWOPolicy (PreTrainedPolicy)
│       └── processor_nwo.py         # Observation/action processing
├── tests/
│   └── test_policy.py       # Unit tests
├── examples/
│   ├── basic_usage.py       # Basic usage example
│   ├── dataset_export.py    # Dataset export example
│   └── README.md
├── pyproject.toml           # Package configuration
├── pytest.ini              # Test configuration
├── MANIFEST.in             # Package manifest
├── README.md               # Main documentation
└── LICENSE                 # MIT License
```

## Quick Start

```bash
# Install
pip install lerobot_policy_nwo

# Set API key
export NWO_API_KEY="your_key_from_nwo.capital"

# Use
from lerobot_policy_nwo import NWOPolicy, NWOPolicyConfig

config = NWOPolicyConfig()
policy = NWOPolicy(config)
action = policy.select_action(observation)
```

## Publishing to PyPI

1. Update version in `pyproject.toml`
2. Create GitHub release
3. GitHub Actions automatically publishes to PyPI

Or manually:
```bash
python -m build
twine upload dist/*
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
