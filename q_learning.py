from dataclasses import dataclass
import numpy as np

from blackjack_env import make_env, evaluate_policy, HIT


@dataclass
class Config:
    episodes: int = 500_000
    alpha: float = 0.01
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.5
    q_init: float = 0.0
    seed: int = 0

    log_every: int = 50_000
    eval_episodes: int = 100_000

    policy_path = "policy.txt"


class QLearningAgent:

    def __init__(self, n_actions=2, alpha=0.01, gamma=1.0,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=250_000,
                 q_init=0.0, seed=0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self.epsilon = epsilon_start
        self.rng = np.random.default_rng(seed)

        self.Q = np.full((32, 11, 2, n_actions), q_init, dtype=np.float64)

    def greedy_action(self, state):
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def update(self, state, action, reward, next_state, terminated):
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = target - self.Q[state + (action,)]
        self.Q[state + (action,)] += self.alpha * td_error
        return td_error

    def decay_epsilon(self, episode):
        frac = min(1.0, episode / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def policy(self, state):
        return int(np.argmax(self.Q[state]))

def train(env, agent, cfg):
    episode_rewards = np.zeros(cfg.episodes)
    epsilons = np.zeros(cfg.episodes)
    state, _ = env.reset(seed=cfg.seed)

    for ep in range(cfg.episodes):
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated)

            total_reward += reward
            state = next_state
            done = terminated or truncated

        episode_rewards[ep] = total_reward
        epsilons[ep] = agent.epsilon
        agent.decay_epsilon(ep + 1)
        state, _ = env.reset()

        if cfg.log_every and (ep + 1) % cfg.log_every == 0:
            recent = episode_rewards[ep + 1 - cfg.log_every: ep + 1].mean()
            print(f"episode {ep + 1:>9,} | epsilon {agent.epsilon:.3f} | "
                  f"avg reward {recent:+.4f}")

    return episode_rewards, epsilons


def save_policy(agent, path):
    dealer_cards = range(1, 11)
    header = "      " + " ".join(f"{'A' if d == 1 else d:>2}" for d in dealer_cards)
    lines = []
    for usable_ace, title in [(0, "Hard"), (1, "Soft")]:
        lines.append(f"{title}")
        lines.append(header)
        for player_sum in range(21, 11, -1):
            row = ["H" if agent.policy((player_sum, d, usable_ace)) == HIT else "S"
                   for d in dealer_cards]
            lines.append(f"  {player_sum:>2}  " + " ".join(f"{a:>2}" for a in row))
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved policy chart to {path}")


def main():
    cfg = Config()
    print("Config:", cfg, "\n")

    env = make_env()
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay_episodes=int(cfg.episodes * cfg.epsilon_decay_fraction),
        q_init=cfg.q_init,
        seed=cfg.seed,
    )

    episode_rewards, epsilons = train(env, agent, cfg)
    save_policy(agent, cfg.policy_path)

    agent_score = evaluate_policy(env, agent.policy, cfg.eval_episodes, seed=cfg.seed + 1)
    print(f"\nevaluation over {cfg.eval_episodes:,} hands:")
    print(f"  mean reward: {agent_score:+.4f}")
    env.close()

if __name__ == "__main__":
    main()