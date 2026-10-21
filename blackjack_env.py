import gymnasium as gym

STAND, HIT = 0, 1

def make_env():
    return gym.make("Blackjack-v1", natural=True, sab=False, render_mode=None)


def evaluate_policy(env, policy, n_episodes=100_000, seed=0):
    total = 0.0
    obs, _ = env.reset(seed=seed)
    for _ in range(n_episodes):
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            done = terminated or truncated
        total += reward
        obs, _ = env.reset()
    return total / n_episodes