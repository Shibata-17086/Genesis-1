import argparse
import os
import pickle

import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init(backend=gs.cpu) #change backend gpu→cpu

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device="cpu", #add device palametor
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir) # delete device pal. cuda:0
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy() # delete device pal. cuda:0

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

# 以下2か所の引数deviceを削除（デフォルトが"cpu"）

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
