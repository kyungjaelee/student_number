import time
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from IPython.display import Video, display

from agent import Agent
from car_racing_env import CarRacingEnv
from gymnasium.vector import AsyncVectorEnv

# === Evaluation Function ===
def evaluate(agent, n_evals=10):
    eval_env = CarRacingEnv()
    total_score = 0
    for eval_idx in range(n_evals):
        s, _ = eval_env.reset(seed=eval_idx)
        done, score = False, 0
        while not done:
            a = agent.select_best_action(s)
            s, r, terminated, truncated, _ = eval_env.step(a)
            score += r
            done = terminated or truncated
        total_score += score
    return np.round(total_score / n_evals, 4)

# === Record Video and Return Score ===
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

# === Environment Factory ===
def make_env():
    def _thunk():
        return CarRacingEnv()
    return _thunk

if __name__ == "__main__":

    num_envs = 10
    num_epochs = 5
    epoch_steps = 1000
    eval_interval = 1  # evaluate every epoch

    parallel_env = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    state_dim = parallel_env.single_observation_space.shape
    action_dim = parallel_env.single_action_space.n

    agent = Agent(state_dim=state_dim, action_dim=action_dim)
    history = {'Epoch': [], 'AvgReturn': []}

    for epoch in range(1, num_epochs + 1):
        agent.train_epoch(parallel_env, epoch_steps)

        if epoch % eval_interval == 0:
            avg_return = evaluate(agent)
            history['Epoch'].append(epoch)
            history['AvgReturn'].append(avg_return)
            print(f"[Epoch {epoch}] Total Steps: {agent.total_steps} | Eval Score: {avg_return}")

    # === Plot and Save Learning Curve ===
    plt.figure(figsize=(6, 4))
    plt.plot(history['Epoch'], history['AvgReturn'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Return')
    plt.title('Training Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.close()

    # === Record and Save Best of 10 Videos ===
    print("Recording 10 runs and saving the best one...")
    best_score = -float('inf')
    best_frames = None
    best_seed = None

    for i in range(10):
        frames, score = record_video_and_score(agent, seed=i)
        if score > best_score:
            best_score = score
            best_frames = frames
            best_seed = i

    os.makedirs("videos", exist_ok=True)
    best_path = f"videos/best_car_racing_seed_{best_seed}.mp4"
    imageio.mimsave(best_path, best_frames, fps=30)
    print(f"Saved best video with return {best_score:.2f} (Seed {best_seed}) to {best_path}")

    display(Video(best_path, embed=True))