"""Training entrypoint.

Example
-------
    python -m rlmalloc.train \
        --dist lognormal_train --episodes 10000 --max-steps 2000 \
        --seed 0 --device cpu \
        --out results/checkpoints/agent \
        --log results/metrics/train_log.csv

Target-network updates happen every ``config.TARGET_UPDATE_FREQ`` *steps*
(paper Alg.1: T = 50).  Epsilon decays once per episode.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

from . import config
from .agent import DQNAgent, save_checkpoint_meta
from .env import MemoryEnv
from .utils import ensure_dir, set_seed, write_csv


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the RLmalloc DQN agent.")
    p.add_argument("--dist", default=config.DEFAULT_TRAIN_DIST,
                   choices=sorted(config.DISTRIBUTIONS))
    p.add_argument("--episodes", type=int, default=config.NUM_EPISODES)
    p.add_argument("--max-steps", type=int, default=config.MAX_STEPS_PER_EPISODE)
    p.add_argument("--learn-every", type=int, default=config.LEARN_EVERY)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--out", default="results/checkpoints/agent")
    p.add_argument("--log", default="results/metrics/train_log.csv")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def train(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)

    env = MemoryEnv(dist=args.dist)
    env.set_python_seed(args.seed)
    agent = DQNAgent(device=args.device, seed=args.seed)

    rows: List[Dict[str, float]] = []
    global_step = 0
    start = time.time()

    try:
        from tqdm import tqdm

        iterator = tqdm(range(args.episodes), desc="train",
                        disable=args.no_progress)
    except Exception:  # pragma: no cover
        iterator = range(args.episodes)

    for episode in iterator:
        state = env.reset()
        episode_return = 0.0
        losses: List[float] = []
        steps = 0

        for _ in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action.item())
            r = float(reward.item())
            d = bool(done.item())

            agent.store_transition(state, action, reward, next_state, done)

            if global_step % args.learn_every == 0:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

            global_step += 1
            if global_step % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            state = next_state
            episode_return += r
            steps += 1
            if d:
                break

        agent.update_epsilon()

        rows.append({
            "episode": episode,
            "return": episode_return,
            "steps": steps,
            "duration": env.duration,
            "epsilon": agent.epsilon,
            "mean_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "occupancy_end": (
                config.MEMORY_SIZE
                - sum(s for _, s in env.free_blocks)
            ) / config.MEMORY_SIZE,
        })

        if not args.no_progress and not hasattr(iterator, "set_postfix"):
            pass

    write_csv(rows, args.log)
    agent.save(args.out)
    meta = {
        "dist": args.dist,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "learn_every": args.learn_every,
        "seed": args.seed,
        "device": args.device,
        "global_steps": global_step,
        "final_epsilon": agent.epsilon,
        "train_seconds": time.time() - start,
        "git_config": {
            "gamma": config.GAMMA,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "replay": config.REPLAY_BUFFER_SIZE,
            "target_update_freq": config.TARGET_UPDATE_FREQ,
        },
    }
    save_checkpoint_meta(args.out, meta)

    last = rows[-1]
    print(f"[train] episodes={args.episodes} steps={global_step} "
          f"final_eps={agent.epsilon:.4f} "
          f"mean_return_last100="
          f"{sum(r['return'] for r in rows[-100:]) / max(len(rows[-100:]), 1):.3f} "
          f"elapsed={meta['train_seconds']:.1f}s")
    print(f"[train] log -> {args.log}")
    print(f"[train] checkpoint -> {args.out}.pt")
    return {"steps": global_step, "seconds": meta["train_seconds"],
            "last_return": last["return"]}


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dir(os.path.dirname(args.log))
    ensure_dir(os.path.dirname(args.out))
    train(args)


if __name__ == "__main__":
    main()
