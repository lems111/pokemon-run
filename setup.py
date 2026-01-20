"""
Setup script for Pokémon Yellow RL project.
Creates required directories and installs dependencies.
"""
import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create required project directories."""
    directories = [
        'env',
        'train',
        'showcase',
        'commentary',
        'overlay/ui',
        'runs/checkpoints',
        'runs/tensorboard',
        'runs/videos',
        'states'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def install_dependencies():
    """Install required Python dependencies."""
    dependencies = [
        'gym',
        'stable-baselines3[extra]',
        'torch',
        'pygame',
        'pillow',
        'requests',
        'numpy',
        'pyboy',
        'aiohttp',
        'python-dotenv'
    ]
    
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def create_requirements_file():
    """Create requirements.txt file."""
    requirements = """
gym
stable-baselines3[extra]
torch
pygame
pillow
requests
numpy
pyboy
aiohttp
python-dotenv
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    print("Created requirements.txt")

def create_readme():
    """Create README.md file."""
    readme_content = """
# Pokémon Yellow RL

Reinforcement Learning agent for Pokémon Yellow using PPO.

## Project Structure

```
.
├── env/                 # Environment files
├── pokemon_yellow_env.py  # Gym-style environment wrapper
├── memory_map.py        # RAM address constants and decoders
├── rewards.py           # Reward shaping functions
├── actions.py           # Action mappings and handling
├── train/               # Training components
│   ├── train_ppo.py     # PPO training loop
│   └── callbacks.py     # Custom training callbacks
├── showcase/            # Live showcase runner
│   └── run_live.py      # Windowed playback with hot reload
├── commentary/          # LLM-based commentary
│   ├── llm_client.py    # LLM API client
│   └── narrator.py      # Prompt builder for commentary
├── overlay/             # OBS overlay
│   └── server.py        # Web server for overlay
├── runs/                # Training outputs
│   ├── checkpoints/     # Model checkpoints
│   ├── tensorboard/     # TensorBoard logs
│   └── videos/        # Optional recordings
└── states/              # Save states
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training:
   ```bash
   python train/train_ppo.py
   ```

3. Run showcase:
   ```bash
   python showcase/run_live.py
   ```

4. Run overlay server:
   ```bash
   python overlay/server.py
   ```

## Features

- Headless PPO training with parallel environments
- Live showcase with windowed playback
- Automatic model checkpoint saving
- LLM-based commentary generation
- OBS browser source overlay
- Curriculum learning with progression stages
- Real-time telemetry and stats display

## Usage

### Training
Run the training script to start training the agent:
```bash
python train/train_ppo.py
```

### Showcase
Run the showcase to see the agent in action with a windowed interface:
```bash
python showcase/run_live.py
```

### Commentary
The system can generate natural-language commentary using local LLMs like Ollama or LM Studio.

### Overlay
The overlay server provides a browser source for OBS that displays real-time game stats and commentary.

## Requirements

- Python 3.7+
- Gym
- Stable-Baselines3
- Pygame
- Torch
- Requests
- Pillow
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content.strip())
    
    print("Created README.md")

def main():
    """Main setup function."""
    print("Setting up Pokémon Yellow RL project...")
    
    # Create directories
    create_directories()
    
    # Create requirements file
    create_requirements_file()
    
    # Create README
    create_readme()
    
    # Install dependencies
    install_dependencies()
    
    print("\nSetup complete! You can now start training and running the agent.")

if __name__ == "__main__":
    main()
