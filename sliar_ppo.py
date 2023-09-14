import os
import torch
import hydra
from hydra.utils import instantiate
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from plotly.subplots import make_subplots
from omegaconf import DictConfig, OmegaConf
import stable_baselines3 as sb3
from stable_baselines3 import PPO, DQN, A2C, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    EventCallback,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
    
    ProgressBarCallback
)
# pip install git+https://github.com/carlosluis/stable-baselines3@cff332c29096e0095ceef20df70be66b1b82d44c

sns.set_theme(style="whitegrid")

class CustomCallback(EventCallback):
    def __init__(
        self,
        eval_env,
        eval_freq
    ):
        super(CustomCallback, self).__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            model = self.locals["self"]
            mean_reward, std_reward = evaluate_policy(model, self.eval_env, n_eval_episodes=1)
            wandb.log({"eval_reward": mean_reward}, step=self.n_calls)

        return continue_training

@hydra.main(version_base=None, config_path="conf", config_name="ppo_sliar")
def main(conf: DictConfig):
    run = wandb.init(project=f"sliar-{conf.exp_name}")
    for k, v in conf.train.items():
        wandb.run.summary[f"train.{k}"] = v
    for k, v in conf.sliar.items():
        wandb.run.summary[f"sliar.{k}"] = v

    train_env = instantiate(conf.sliar)
    check_env(train_env)
    log_dir = "./sliar_ppo_log"
    os.makedirs(log_dir, exist_ok=True)
    train_env = Monitor(train_env, log_dir)
    policy_kwargs = dict(
                            # activation_fn=torch.nn.ReLU,
                            # net_arch=[256, 128, 64]
                        )
    Algorithm = getattr(sb3, conf.train.algorithm)
    model = Algorithm(  "MlpPolicy", train_env,
                        clip_range=conf.train.clip_range,
                        policy_kwargs=policy_kwargs)
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=10)
    print("Before:")
    print(f"\tmean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    eval_env = instantiate(conf.sliar)
    eval_callback = EvalCallback(
            eval_env,
            eval_freq=1000,
            verbose=0,
            warn=False,
            log_path='eval_log',
            best_model_save_path='best_model'
        )
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/',
                                             name_prefix=f"rl-{conf.train.algorithm}")
    callback = CallbackList([checkpoint_callback, eval_callback, CustomCallback(eval_env=eval_env, eval_freq=100), ProgressBarCallback()])

    model.learn(total_timesteps=conf.train.n_steps, callback=callback)
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=10)
    print("After:")
    print(f"\tmean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    os.makedirs('figures', exist_ok=True)
    df = pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)
    sns.lineplot(data=df.r)
    plt.xlabel('episodes')
    plt.ylabel('The cummulative return')
    plt.savefig(f"figures/reward.png")
    # wandb.log({"reward history": plt})
    plt.close()

    image = wandb.Image(f"figures/reward.png", caption="Rewards History")
    wandb.log({"Pic History": image})

    # Visualize Controlled sliar Dynamics
    model = Algorithm.load(f'best_model/best_model.zip')
    state = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, _, done, _, _ = eval_env.step(action)

    df = eval_env.dynamics
    best_reward = df.rewards.sum()
    plt.figure(figsize=(8,8))
    plt.subplot(5, 1, 1)
    plt.title(f"R = {df.rewards.sum():,.4f}")
    sns.lineplot(data=df, x='days', y='infected', color='r')
    plt.xticks(color='w')
    plt.subplot(5, 1, 2)
    sns.lineplot(data=df, x='days', y='nus', color='k', drawstyle='steps-pre')
    plt.ylim([-0.001, max(conf.sliar.nu_max * 1.1, 0.01)])
    plt.xticks(color='w')
    plt.subplot(5, 1, 3)
    sns.lineplot(data=df, x='days', y='taus', color='b', drawstyle='steps-pre')
    plt.ylim([-0.001, max(conf.sliar.tau_max * 1.1, 0.01)])
    plt.xticks(color='w')
    plt.subplot(5, 1, 4)
    sns.lineplot(data=df, x='days', y='sigmas', color='orange', drawstyle='steps-pre')
    plt.ylim([-0.001, max(conf.sliar.sigma_max * 1.1, 0.01)])
    plt.xticks(color='w')
    plt.subplot(5, 1, 5)
    sns.lineplot(data=df, x='days', y='rewards', color='g')
    plt.savefig(f"figures/best.png")
    plt.close()

    best_checkpoint = ""
    max_val = -float('inf')
    for path in tqdm(os.listdir('checkpoints')):
        model = Algorithm.load(f'checkpoints/{path}')
        state, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, _, done, _, _ = eval_env.step(action)
        df = eval_env.dynamics

        cum_reward = df.rewards.sum()
        if cum_reward > max_val:
            max_val = cum_reward
            best_checkpoint = path

        plt.figure(figsize=(8,8))
        plt.subplot(5, 1, 1)
        plt.title(f"R = {df.rewards.sum():,.4f}")
        sns.lineplot(data=df, x='days', y='infected', color='r')
        plt.xticks(color='w')
        plt.subplot(5, 1, 2)
        sns.lineplot(data=df, x='days', y='nus', color='k', drawstyle='steps-pre')
        plt.ylim([-0.001, max(conf.sliar.nu_max * 1.1, 0.01)])
        plt.xticks(color='w')
        plt.subplot(5, 1, 3)
        sns.lineplot(data=df, x='days', y='taus', color='b', drawstyle='steps-pre')
        plt.ylim([-0.001, max(conf.sliar.tau_max * 1.1, 0.01)])
        plt.xticks(color='w')
        plt.subplot(5, 1, 4)
        sns.lineplot(data=df, x='days', y='sigmas', color='orange', drawstyle='steps-pre')
        plt.ylim([-0.001, max(conf.sliar.sigma_max * 1.1, 0.01)])
        plt.xticks(color='w')
        plt.subplot(5, 1, 5)
        sns.lineplot(data=df, x='days', y='rewards', color='g')
        plt.savefig(f"figures/{path.replace('.zip', '.png')}")
        plt.close()



    # Visualize Controlled sliar Dynamics
    if best_reward < max_val:
        best_reward = max_val
        model = Algorithm.load(f'checkpoints/{best_checkpoint}')
        state, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, _, done, _, _ = eval_env.step(action)
        df = eval_env.dynamics
        # sns.lineplot(data=df, x='days', y='susceptible')
        plt.figure(figsize=(8,8))
        plt.subplot(5, 1, 1)
        plt.title(f"R = {df.rewards.sum():,.4f}")
        sns.lineplot(data=df, x='days', y='infected', color='r')
        plt.xticks(color='w')
        plt.subplot(5, 1, 2)
        sns.lineplot(data=df, x='days', y='nus', color='k', drawstyle='steps-pre')
        plt.ylim([-0.001, max(conf.sliar.nu_max * 1.1, 0.01)])
        plt.xticks(color='w')
        plt.subplot(5, 1, 3)
        sns.lineplot(data=df, x='days', y='taus', color='b', drawstyle='steps-pre')
        plt.ylim([-0.001, max(conf.sliar.tau_max * 1.1, 0.01)])
        plt.xticks(color='w')
        plt.subplot(5, 1, 4)
        sns.lineplot(data=df, x='days', y='sigmas', color='orange', drawstyle='steps-pre')
        plt.ylim([-0.001, max(conf.sliar.sigma_max * 1.1, 0.01)])
        plt.xticks(color='w')
        plt.subplot(5, 1, 5)
        sns.lineplot(data=df, x='days', y='rewards', color='g')
        plt.savefig(f"figures/best.png")
        # wandb.log({f"R = {df.rewards.sum():,.4f}": plt})
        plt.close()

    image = wandb.Image(f"figures/best.png", caption=f"{best_checkpoint}")
    wandb.log({"Pic Best Reward": image})
    wandb.run.summary["best reward"] = best_reward

    run.finish()

if __name__ == '__main__':
    main()

# python sliar_ppo.py train.n_steps=1500000 sliar.continuous=false train.clip_range=0.01,0.05,0.1,0.2,0.5
# python sliar_ppo.py train.n_steps=1500000 sliar.continuous=true train.clip_range=0.01,0.05,0.1,0.2,0.5
# python sliar_ppo.py train.n_steps=1500000 sliar.continuous=true train.clip_range=0.1 sliar.Q=0.01,0.025,0.05,0.1