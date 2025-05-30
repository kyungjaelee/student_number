# test.py

import os
import argparse
import yaml
import json
import numpy as np
import imageio
import torch
from IPython.display import Video, display

from agent import Agent
from car_racing_env import CarRacingEnv
from exploration import EpsilonGreedy, SoftmaxExploration

# === Load config ===
def load_config(path):
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        elif path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported config format")

# === Build exploration strategy ===
EXPLORATION_MAP = {
    "EpsilonGreedy": EpsilonGreedy,
    "SoftmaxExploration": SoftmaxExploration,
}

def build_exploration_strategy(cfg, action_dim):
    name = cfg.get("name", "SoftmaxExploration")
    params = cfg.get("params", {})
    cls = EXPLORATION_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown exploration strategy: {name}")
    return cls(action_dim=action_dim, **params)

# === Record video ===
def record_video_and_score(agent, seed):
    env = CarRacingEnv(render_mode='rgb_array')
    env.reset(seed=seed)
    s, _ = env.reset()
    frames = []
    done = False
    score = 0

    while not done:
        frame = env.render()
        frames.append(frame)
        a = agent.select_best_action(s)
        s, r, terminated, truncated, _ = env.step(a)
        score += r
        done = terminated or truncated

    env.close()
    return frames, score

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained agent with config")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Path to saved model")
    args = parser.parse_args()

    config = load_config(args.config)
    agent_cfg = config["agent"]

    # Dummy env to get dimensions
    dummy_env = CarRacingEnv()
    state_dim = dummy_env.observation_space.shape
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    # Rebuild agent from config
    exploration_strategy = build_exploration_strategy(agent_cfg.get("exploration_strategy", {}), action_dim)

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        exploration_strategy=exploration_strategy,
        lr=agent_cfg.get("lr", 0.00025),
        gamma=agent_cfg.get("gamma", 0.99),
        batch_size=agent_cfg.get("batch_size", 64),
        warmup_steps=agent_cfg.get("warmup_steps", 5000),
        buffer_size=agent_cfg.get("buffer_size", int(1e5)),
        target_update_interval=agent_cfg.get("target_update_interval", 5000),
        use_double_q=agent_cfg.get("use_double_q", False),
        use_per=agent_cfg.get("use_per", False)
    )

    # Load trained model
    best_model = torch.load(args.model_path, map_location=agent.device)
    agent.network.load_state_dict(best_model)
    agent.network.eval()

    print("Recording 10 runs and saving the best one...")
    # --- Record and save all videos sequentially ---
    all_frames = []
    for i in range(10):
        frames, score = record_video_and_score(agent, seed=i)
        print(f"Seed {i} return: {score:.2f}")
        all_frames.extend(frames)  # ✅ append all frames

    merged_path = "videos/all_runs.mp4"
    imageio.mimsave(merged_path, all_frames, fps=30)
    print(f"Saved merged video to {merged_path}")

    display(Video(merged_path, embed=True))