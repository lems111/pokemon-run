# How to Watch the AI Model Playing Pokémon Yellow

This guide explains how to watch a trained AI model play Pokémon Yellow using the showcase functionality.

## Prerequisites

Make sure you have:
- Pokémon Yellow ROM (`pokemon_yellow.gb`) in the project root
- Save states in the `states/` directory
- Trained model checkpoints in `runs/checkpoints/`

## Running the Showcase

To watch the AI model play Pokémon Yellow, run:

```bash
python3 showcase/run_live.py
```

This will:
1. Load the latest trained model from `runs/checkpoints/`
2. Start the Pokémon Yellow emulator in a window
3. Show the AI playing the game using the trained policy
4. Display real-time statistics like steps taken, tiles visited, and position

## Features

- **Real-time Model Loading**: The showcase will automatically load the latest checkpoint when available
- **Live Statistics**: See the AI's progress in real-time
- **Position Tracking**: Watch where the AI is moving in the game world
- **Model Hot-swapping**: If new checkpoints are saved during runtime, the showcase will automatically switch to the newer model

## Model Checkpoints

The project contains trained models at different training steps:
- `runs/checkpoints/ppo_10000.zip` - 10,000 training steps
- `runs/checkpoints/ppo_20000.zip` - 20,000 training steps

The showcase will automatically load the latest model available.

## Troubleshooting

If you encounter issues:
1. Make sure `pokemon_yellow.gb` is in the project root
2. Ensure you have the required dependencies installed (`pip install -r requirements.txt`)
3. Check that save states exist in the `states/` directory
4. Verify that the model files in `runs/checkpoints/` are not corrupted

## Custom Model Path

To use a specific model checkpoint, modify the `model_path` parameter in `showcase/run_live.py`:

```python
runner = ShowcaseRunner(
    model_path="runs/checkpoints/ppo_10000.zip",  # Use specific model
    save_state_path="states/start_game.state",
    window_size=(800, 600),
    fps=60
)
```