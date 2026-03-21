"""Vectorized environment using multiprocessing for parallel MuJoCo rollouts."""

import multiprocessing as mp
import numpy as np


def _worker(remote, parent_remote, env_fn):
    """Worker process that runs a single environment."""
    parent_remote.close()
    env = env_fn()
    available_faces = [1]

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                info["terminal_obs"] = obs
                target_face = int(np.random.choice(available_faces))
                obs = env.reset(target_face=target_face)
            remote.send((obs, reward, done, info))

        elif cmd == "reset":
            obs = env.reset(**data)
            remote.send(obs)

        elif cmd == "set_faces":
            available_faces = list(data)
            remote.send(None)

        elif cmd == "set_curriculum_angle":
            env.set_curriculum_max_angle(float(data))
            remote.send(None)

        elif cmd == "set_start_faces":
            env.set_curriculum_start_faces(list(data))
            remote.send(None)

        elif cmd == "set_max_episode_steps":
            env.max_episode_steps = int(data)
            remote.send(None)

        elif cmd == "set_action_scale":
            env.set_action_scale(float(data))
            remote.send(None)

        elif cmd == "set_reward_config":
            env.set_reward_config(data)
            remote.send(None)

        elif cmd == "set_continuous_episodes":
            env.set_continuous_episodes(bool(data))
            remote.send(None)

        elif cmd == "get_attr":
            remote.send(getattr(env, data))

        elif cmd == "close":
            env.close()
            remote.close()
            break


class SubprocVecEnv:
    """Vectorized environment that runs multiple envs in subprocesses."""

    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.waiting = False

        # Create pipes
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])

        # Start workers
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = mp.Process(target=_worker, args=(work_remote, remote, env_fn), daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get env attributes
        self.remotes[0].send(("get_attr", "obs_dim"))
        self.obs_dim = self.remotes[0].recv()
        self.remotes[0].send(("get_attr", "act_dim"))
        self.act_dim = self.remotes[0].recv()

    def set_available_faces(self, faces):
        """Set which faces are available for auto-reset."""
        for remote in self.remotes:
            remote.send(("set_faces", faces))
        # Receive acknowledgments
        for remote in self.remotes:
            remote.recv()

    def set_start_faces(self, start_faces):
        """Set which faces the dice can start on in all envs."""
        for remote in self.remotes:
            remote.send(("set_start_faces", start_faces))
        for remote in self.remotes:
            remote.recv()

    def set_curriculum_angle(self, max_angle):
        """Set max angular distance from target for start orientation in all envs."""
        for remote in self.remotes:
            remote.send(("set_curriculum_angle", max_angle))
        for remote in self.remotes:
            remote.recv()

    def set_max_episode_steps(self, steps):
        """Set max episode steps in all envs."""
        for remote in self.remotes:
            remote.send(("set_max_episode_steps", steps))
        for remote in self.remotes:
            remote.recv()

    def set_action_scale(self, scale):
        """Set action scale in all envs."""
        for remote in self.remotes:
            remote.send(("set_action_scale", scale))
        for remote in self.remotes:
            remote.recv()

    def set_reward_overrides(self, overrides):
        """Update reward config in all envs with per-phase overrides."""
        for remote in self.remotes:
            remote.send(("set_reward_config", overrides))
        for remote in self.remotes:
            remote.recv()

    def set_continuous_episodes(self, enabled):
        """Toggle continuous episodes in all envs."""
        for remote in self.remotes:
            remote.send(("set_continuous_episodes", enabled))
        for remote in self.remotes:
            remote.recv()

    def reset_all(self, target_faces=None):
        """Reset all environments. Returns (n_envs, obs_dim) array."""
        if target_faces is None:
            target_faces = [None] * self.n_envs

        for remote, tf in zip(self.remotes, target_faces):
            kwargs = {"target_face": tf} if tf is not None else {}
            remote.send(("reset", kwargs))

        obs = np.stack([remote.recv() for remote in self.remotes])
        return obs

    def step(self, actions):
        """Step all envs. actions shape: (n_envs, act_dim).

        Returns:
            obs: (n_envs, obs_dim) - next obs (auto-reset if done)
            rewards: (n_envs,)
            dones: (n_envs,)
            infos: list of dicts
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)

        return np.stack(obs), np.array(rewards), np.array(dones), list(infos)

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for process in self.processes:
            process.join(timeout=5)


def make_env_fn(xml_path, env_config, seed=None):
    """Create a closure that builds an env (for multiprocessing)."""
    def _make():
        from envs.dex_cube_env import DexCubeEnv
        env = DexCubeEnv(xml_path=xml_path, config=env_config)
        env._available_faces = [1]  # default
        if seed is not None:
            np.random.seed(seed)
        return env
    return _make
