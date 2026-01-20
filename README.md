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

### Metrics
```bash
tensorboard --logdir=runs/tensorboard
```

## Requirements

- Python 3.7+
- Gym
- Stable-Baselines3
- Pygame
- Torch
- Requests
- Pillow
- pyboy