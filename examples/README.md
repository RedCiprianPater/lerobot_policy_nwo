# LeRobot Policy NWO - Examples

This directory contains example scripts showing how to use the NWO Robotics adapter with LeRobot.

## Examples

### 1. Basic Usage (`basic_usage.py`)

Shows how to:
- Initialize the NWO policy
- Get actions from observations
- Use in a control loop
- Save/load policy config

```bash
export NWO_API_KEY="your_api_key_here"
python examples/basic_usage.py
```

### 2. Dataset Export (`dataset_export.py`)

Shows how to:
- Export NWO API usage as training dataset
- Convert to LeRobot format
- Prepare for imitation learning

```bash
export NWO_API_KEY="your_api_key_here"
python examples/dataset_export.py
```

## Integration with Real Robots

To use with a real robot, replace the `simulate_robot_observation()` function with actual sensor readings:

```python
def get_robot_observation():
    # Example with RealSense camera and robot arm
    import pyrealsense2 as rs
    from robot_controller import get_joint_positions
    
    # Get camera image
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    image = np.asanyarray(color_frame.get_data())
    
    # Convert to tensor (3, 224, 224)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1) / 255.0
    
    # Get joint positions
    joints = get_joint_positions()  # Your robot's API
    joint_tensor = torch.tensor(joints)
    
    return {
        "observation.image": image_tensor,
        "observation.state": joint_tensor,
    }
```

## LeRobot Training Workflow

1. **Collect data with NWO policy** (this adapter)
2. **Export dataset** from NWO dashboard or API
3. **Train with LeRobot**:
   ```bash
   python lerobot/scripts/train.py \
     --dataset.dir=path/to/nwo_dataset \
     --policy.type=diffusion \
     --output_dir=./my_policy
   ```
4. **Deploy trained policy** on your robot

## More Resources

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [NWO Robotics](https://nwo.capital/webapp/nwo-robotics.html)
- [API Dashboard](https://nwo.capital/webapp/api-key.php)
