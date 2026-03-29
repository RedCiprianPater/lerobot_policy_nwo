#!/usr/bin/env python3
"""Example: Using NWO Policy with LeRobot for robot control.

This example shows how to:
1. Initialize the NWO policy
2. Convert observations to actions
3. Use with a simulated or real robot
"""

import os
import torch
import numpy as np

# Set your API key (get one at https://nwo.capital/webapp/api-key.php)
os.environ["NWO_API_KEY"] = "your_api_key_here"

from lerobot_policy_nwo import NWOPolicy, NWOPolicyConfig


def simulate_robot_observation():
    """Simulate getting an observation from a robot.
    
    In real usage, this would come from your robot's sensors.
    """
    # Simulated camera image (3, 224, 224) - normalized to [0, 1]
    image = torch.rand(3, 224, 224)
    
    # Simulated joint positions (6 joints)
    joint_positions = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])
    
    return {
        "observation.image": image,
        "observation.state": joint_positions,
    }


def main():
    print("=" * 60)
    print("NWO Robotics Policy - LeRobot Adapter Example")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("NWO_API_KEY", "")
    if api_key == "your_api_key_here" or not api_key:
        print("\n⚠️  Please set your NWO_API_KEY environment variable!")
        print("Get your free API key at: https://nwo.capital/webapp/api-key.php")
        print("\nExiting...")
        return
    
    # Initialize policy
    print("\n1. Initializing NWO Policy...")
    config = NWOPolicyConfig(
        instruction_template="Execute robot manipulation task",
        use_image=True,
    )
    policy = NWOPolicy(config)
    print("   ✓ Policy initialized")
    
    # Health check
    print("\n2. Checking API health...")
    health = policy.health_check()
    if health.get("status") == "ok":
        print(f"   ✓ API is healthy (version: {health.get('version', 'unknown')})")
    else:
        print(f"   ⚠️  API health check failed: {health}")
    
    # Simulate robot control loop
    print("\n3. Running simulated control loop...")
    print("-" * 60)
    
    tasks = [
        "Pick up the red cube",
        "Place it on the blue platform",
        "Move arm to home position",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n   Step {i}: {task}")
        
        # Get observation from robot
        observation = simulate_robot_observation()
        
        # Get action from NWO policy
        try:
            action = policy.select_action(observation, task_description=task)
            
            # Get metadata from the API call
            metadata = policy.get_last_metadata()
            
            print(f"      Action shape: {action.shape}")
            print(f"      Confidence: {metadata['confidence']:.2%}")
            print(f"      Request ID: {metadata['request_id']}")
            print(f"      Action values: {action[0].tolist()}")
            
        except Exception as e:
            print(f"      ✗ Error: {e}")
    
    print("\n" + "-" * 60)
    print("\n4. Example complete!")
    
    # Show how to save/load policy
    print("\n5. Saving policy config...")
    policy.save_pretrained("./saved_nwo_policy")
    print("   ✓ Saved to ./saved_nwo_policy/")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  - Replace simulate_robot_observation() with real robot code")
    print("  - Use policy.select_action() in your control loop")
    print("  - Collect data for LeRobot training with dataset export")
    print("=" * 60)


if __name__ == "__main__":
    main()
