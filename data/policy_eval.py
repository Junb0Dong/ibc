# coding=utf-8
# Copyright 2024 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates TF-Agents policies."""
import functools
import os
import shutil

# 添加Zarr处理工具类
import zarr
import numpy as np
from typing import List, Dict, Any
import json
from tf_agents.trajectories import StepType  # 新增这一行

from absl import app
from absl import flags
from absl import logging

import gin
# Need import to get env resgistration.
from ibc.environments.block_pushing import block_pushing  # pylint: disable=unused-import
from ibc.environments.block_pushing import block_pushing_discontinuous
from ibc.environments.block_pushing import block_pushing_multimodal
from ibc.environments.collect.utils import get_oracle as get_oracle_module
from ibc.environments.particle import particle  # pylint: disable=unused-import
from ibc.environments.particle import particle_oracles
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.metrics import py_metrics
# Need import to get tensorflow_probability registration.
from tf_agents.policies import greedy_policy  # pylint: disable=unused-import
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import example_encoding_dataset


flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

flags.DEFINE_integer('num_episodes', 5, 'Number of episodes to evaluate.')
flags.DEFINE_integer('history_length', None,
                     'If set the previous n observations are stacked.')
flags.DEFINE_bool('video', False,
                  'If true record a video of the evaluations.')
flags.DEFINE_bool('viz_img', False,
                  'If true records an img of evaluation trajectories.')
flags.DEFINE_string('output_path', '/tmp/ibc/policy_eval/',
                    'Path to save videos at.')
flags.DEFINE_enum(
    'task', None,
    ['REACH', 'PUSH', 'INSERT', 'REACH_NORMALIZED', 'PUSH_NORMALIZED',
     'PARTICLE', 'PUSH_DISCONTINUOUS', 'PUSH_MULTIMODAL'],
    'Which task of the enum to evaluate.')
flags.DEFINE_bool('use_image_obs', False,
                  'Whether to include image observations.')
flags.DEFINE_bool('flatten_env', False,
                  'If True the environment observations are flattened.')
flags.DEFINE_bool('shared_memory', False,
                  'If True the connection to pybullet uses shared memory.')
flags.DEFINE_string('saved_model_path', None,
                    'Path to the saved_model policy to eval.')
flags.DEFINE_string('checkpoint_path', None,
                    'Path to the checkpoint to evaluate.')
flags.DEFINE_enum('policy', None, [
    'random', 'oracle_reach', 'oracle_push', 'oracle_reach_normalized',
    'oracle_push_normalized', 'particle_green_then_blue'
], 'Static policies to evaluate.')
flags.DEFINE_string(
    'dataset_path', None,
    'If set a dataset of the policy evaluation will be saved '
    'to the given path.')
flags.DEFINE_integer('replicas', None,
                     'Number of parallel replicas generating evaluations.')


class ZarrDatasetWriter:
    def __init__(self, root_path: str, obs_spec: Dict, action_spec: Dict):
        """初始化Zarr数据集写入器
        
        Args:
            root_path: Zarr根目录路径
            obs_spec: 观测数据的结构描述
            action_spec: 动作数据的结构描述
        """
        self.root = zarr.open(root_path, mode='w')
        
        # 创建数据目录结构
        self.data = self.root.create_group('data')
        self.meta = self.root.create_group('meta')
        
        # 初始化episode元数据列表
        self.episodes = []
        self.current_episode = None
        self.total_steps = 0
        
        # 根据数据规格初始化数组
        self._init_arrays(obs_spec, action_spec)

    def _init_arrays(self, obs_spec: Dict, action_spec: Dict):
        """根据数据规格初始化Zarr数组"""
        # 初始化观测数据数组
        # self.obs_group = self.data.create_group('observations')
        for key, spec in obs_spec.items():
            shape = (0,) + tuple(spec.shape)  # 第一个维度为时间步，动态扩展
            # 兼容 tf dtypes
            dtype = getattr(spec, 'dtype', None)
            if hasattr(dtype, 'as_numpy_dtype'):
                dtype = dtype.as_numpy_dtype
            self.data.create_dataset(
                key,
                shape=shape,
                dtype=dtype,
                chunks=(100,) + tuple(spec.shape),  # 分块设置
                compressor=zarr.Blosc(cname='zstd', clevel=3)  # 压缩设置
            )

        # 初始化动作数据数组
        action_shape = (0,) + tuple(action_spec.shape)
        action_dtype = getattr(action_spec, 'dtype', None)
        if hasattr(action_dtype, 'as_numpy_dtype'):
            action_dtype = action_dtype.as_numpy_dtype
        self.data.create_dataset(
            'actions',
            shape=action_shape,
            dtype=action_dtype,
            chunks=(100,) + tuple(action_spec.shape),
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )

        # 初始化奖励和折扣数组
        self.data.create_dataset(
            'rewards',
            shape=(0,),
            dtype=np.float32,
            chunks=(100,),
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )

        self.data.create_dataset(
            'discounts',
            shape=(0,),
            dtype=np.float32,
            chunks=(100,),
            compressor=zarr.Blosc(cname='zstd', clevel=3)
        )

    def start_episode(self):
        """开始新的episode记录"""
        self.current_episode = {
            'start_idx': self.total_steps,
            'end_idx': None,
            'length': 0,
            'total_reward': 0.0
        }

    def add_step(self, observation: Dict, action: np.ndarray, reward: float, discount: float):
        """添加一步数据"""
        if self.current_episode is None:
            self.start_episode()

        # 扩展并写入观测数据
        for key, value in observation.items():
            arr = self.data[key]
            arr.resize(arr.shape[0] + 1, *arr.shape[1:])
            arr[-1] = np.asarray(value)

        # 扩展并写入动作
        actions = self.data['actions']
        actions.resize(actions.shape[0] + 1, *actions.shape[1:])
        actions[-1] = np.asarray(action)

        # 扩展并写入 reward / discount (一维标量序列)
        rewards = self.data['rewards']
        rewards.resize(rewards.shape[0] + 1)
        rewards[-1] = float(reward)

        discounts = self.data['discounts']
        discounts.resize(discounts.shape[0] + 1)
        discounts[-1] = float(discount)

        # 更新episode信息
        self.current_episode['length'] += 1
        self.current_episode['total_reward'] += float(reward)
        self.total_steps += 1

    def end_episode(self):
        """结束当前episode记录"""
        if self.current_episode is not None:
            self.current_episode['end_idx'] = self.total_steps
            self.episodes.append(self.current_episode)
            self.current_episode = None

    def finalize(self):
        """完成数据集写入，保存元数据"""
        # 如果还有未结束的 episode，先结束它以保证 meta 正确
        if self.current_episode is not None:
            self.end_episode()
        # 基本统计信息作为 attrs
        self.meta.attrs['total_episodes'] = len(self.episodes)
        self.meta.attrs['total_steps'] = self.total_steps
        self.meta.attrs['creation_time'] = str(np.datetime64('now'))

        # 把 episode_ends / lengths / total_rewards 写成可直接读取的数组到 meta group
        try:
            end_indices = np.array([ep['end_idx'] for ep in self.episodes], dtype=np.int64) \
                if len(self.episodes) > 0 else np.zeros((0,), dtype=np.int64)
            lengths = np.array([ep.get('length', 0) for ep in self.episodes], dtype=np.int32) \
                if len(self.episodes) > 0 else np.zeros((0,), dtype=np.int32)
            total_rewards = np.array([ep.get('total_reward', 0.0) for ep in self.episodes], dtype=np.float32) \
                if len(self.episodes) > 0 else np.zeros((0,), dtype=np.float32)
            # 如果已存在则覆盖
            if 'episode_ends' in self.meta:
                del self.meta['episode_ends']
            if 'episode_lengths' in self.meta:
                del self.meta['episode_lengths']
            if 'episode_total_rewards' in self.meta:
                del self.meta['episode_total_rewards']

            self.meta.create_dataset('episode_ends', data=end_indices, dtype=end_indices.dtype)
            self.meta.create_dataset('episode_lengths', data=lengths, dtype=lengths.dtype)
            self.meta.create_dataset('episode_total_rewards', data=total_rewards, dtype=total_rewards.dtype)
        except Exception:
            logging.exception('Failed to write episode arrays into meta.')

        # 仍然保留完整 episodes 结构的 JSON 备份（写入文件或 attrs）
        try:
            episodes_path = None
            if hasattr(self.root, 'store') and hasattr(self.root.store, 'path'):
                episodes_path = os.path.join(self.root.store.path, 'episodes.json')

            if episodes_path:
                with open(episodes_path, 'w') as f:
                    json.dump(self.episodes, f)
            else:
                # 如果不能写文件，则把 JSON 存到 attrs（注意长度限制）
                self.meta.attrs['episodes_json'] = json.dumps(self.episodes)
        except Exception:
            logging.exception('Failed to persist episodes JSON.')



def evaluate(num_episodes,
             task,
             use_image_obs,
             shared_memory,
             flatten_env,
             saved_model_path=None,
             checkpoint_path=None,
             static_policy=None,
             dataset_path=None,
             history_length=None,
             video=False,
             viz_img=False,
             output_path=None):
  """Evaluates the given policy for n episodes."""
  if task in ['REACH', 'PUSH', 'INSERT', 'REACH_NORMALIZED', 'PUSH_NORMALIZED']:
    # Options are supported through flags to build_env_name, and different
    # registered envs.
    env_name = block_pushing.build_env_name(task, shared_memory, use_image_obs)
  elif task in ['PUSH_DISCONTINUOUS']:
    env_name = block_pushing_discontinuous.build_env_name(
        task, shared_memory, use_image_obs)
  elif task in ['PUSH_MULTIMODAL']:
    env_name = block_pushing_multimodal.build_env_name(
        task, shared_memory, use_image_obs)
  elif task == 'PARTICLE':
    # Options are supported through gin, registered env is the same.
    env_name = 'Particle-v0'
    assert not (shared_memory or use_image_obs)  # Not supported.
  else:
    raise ValueError("I don't recognize this task to eval.")

  if flatten_env:
    env = suite_gym.load(
        env_name, env_wrappers=[wrappers.FlattenObservationsWrapper])
  else:
    env = suite_gym.load(env_name)

  if history_length:
    env = wrappers.HistoryWrapper(
        env, history_length=history_length, tile_first_step_obs=True)

  if video:
    video_path = output_path

    if saved_model_path:
      policy_name = os.path.basename(os.path.normpath(saved_model_path))
      checkpoint_ref = checkpoint_path.split('_')[-1]
      video_path = os.path.join(video_path,
                                policy_name + '_' + checkpoint_ref + 'vid.mp4')

    if static_policy:
      video_path = os.path.join(video_path, static_policy, 'vid.mp4')


  if saved_model_path and static_policy:
    raise ValueError(
        'Only pass in either a `saved_model_path` or a `static_policy`.')

  if saved_model_path:
    if not checkpoint_path:
      raise ValueError('Must provide a `checkpoint_path` with a saved_model.')
    policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        saved_model_path, load_specs_from_pbtxt=True)
    policy.update_from_checkpoint(checkpoint_path)
  else:
    if static_policy == 'random':
      policy = random_py_policy.RandomPyPolicy(env.time_step_spec(),
                                               env.action_spec())
    elif task == 'PARTICLE':
      if static_policy == 'particle_green_then_blue':
        # TODO(peteflorence): support more particle oracle options.
        policy = particle_oracles.ParticleOracle(env)
      else:
        raise ValueError('Unknown policy for given task: %s: ' % static_policy)
    elif task != 'PARTICLE':
      # Get an oracle.
      policy = get_oracle_module.get_oracle(env, flags.FLAGS.task)
    else:
      raise ValueError('Unknown policy: %s: ' % static_policy)

  metrics = [
      py_metrics.AverageReturnMetric(buffer_size=num_episodes),
      py_metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes),
  ]
  env_metrics, success_metric = env.get_metrics(num_episodes)
  metrics += env_metrics

  observers = metrics[:]

  if viz_img and ('Particle' in env_name):
    visualization_dir = '/home/ps/tmp/particle_oracle'
    shutil.rmtree(visualization_dir, ignore_errors=True)
    env.set_img_save_dir(visualization_dir)
    observers += [env.save_image]

  if dataset_path:
    # TODO(oars, peteflorence): Consider a custom observer to filter only
    # positive examples.
    # observers.append(
    #     example_encoding_dataset.TFRecordObserver(
    #         dataset_path,
    #         policy.collect_data_spec,
    #         py_mode=True,
    #         compress_image=True))
    # 获取数据规格
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    
    # 创建Zarr写入器
    zarr_writer = ZarrDatasetWriter(dataset_path, obs_spec, action_spec)


    class ZarrObserver:
        def __init__(self, writer):
            self.writer = writer
            self.episode_started = False
        
        def __call__(self, traj):
            if traj.step_type == StepType.FIRST:
                self.writer.start_episode()
                self.episode_started = True
                
            
            # 记录步骤数据
            self.writer.add_step(
                observation=traj.observation,
                action=traj.action,
                reward=traj.reward,
                discount=traj.discount
            )
            
            if traj.step_type == StepType.LAST:
                self.writer.end_episode()
                self.episode_started = False
    
    observers.append(ZarrObserver(zarr_writer))

  driver = py_driver.PyDriver(env, policy, observers, max_episodes=num_episodes)
  time_step = env.reset()
  initial_policy_state = policy.get_initial_state(1)
  driver.run(time_step, initial_policy_state)
  log = ['{0} = {1}'.format(m.name, m.result()) for m in metrics]
  logging.info('\n\t\t '.join(log))

  if zarr_writer:
    zarr_writer.finalize()

  env.close()


def main(_):
  logging.set_verbosity(logging.INFO)
  gin.add_config_file_search_path(os.getcwd())
  gin.parse_config_files_and_bindings(flags.FLAGS.gin_file,
                                      flags.FLAGS.gin_bindings)

  if flags.FLAGS.replicas:
    jobs = []
    if not flags.FLAGS.dataset_path:
      raise ValueError(
          'A dataset_path must be provided when replicas are specified.')
    dataset_split_path = os.path.splitext(flags.FLAGS.dataset_path)
    context = multiprocessing.get_context()

    for i in range(flags.FLAGS.replicas):
      print("dataset_split_path[0]: ", dataset_split_path[0])
      print("dataset_split_path[1]: ", dataset_split_path[1])
      # dataset_path = dataset_split_path[0] + '_%d' % i + dataset_split_path[1]
      dataset_path = dataset_split_path[0] + dataset_split_path[1]
      kwargs = dict(
          num_episodes=flags.FLAGS.num_episodes,
          task=flags.FLAGS.task,
          use_image_obs=flags.FLAGS.use_image_obs,
          shared_memory=flags.FLAGS.shared_memory,
          flatten_env=flags.FLAGS.flatten_env,
          saved_model_path=flags.FLAGS.saved_model_path,
          checkpoint_path=flags.FLAGS.checkpoint_path,
          static_policy=flags.FLAGS.policy,
          dataset_path=dataset_path,
          history_length=flags.FLAGS.history_length
      )
      job = context.Process(target=evaluate, kwargs=kwargs)  # pytype: disable=attribute-error  # re-none
      job.start()
      jobs.append(job)

    for job in jobs:
      job.join()

  else:
    evaluate(
        num_episodes=flags.FLAGS.num_episodes,
        task=flags.FLAGS.task,
        use_image_obs=flags.FLAGS.use_image_obs,
        shared_memory=flags.FLAGS.shared_memory,
        flatten_env=flags.FLAGS.flatten_env,
        saved_model_path=flags.FLAGS.saved_model_path,
        checkpoint_path=flags.FLAGS.checkpoint_path,
        static_policy=flags.FLAGS.policy,
        dataset_path=flags.FLAGS.dataset_path,
        history_length=flags.FLAGS.history_length,
        video=flags.FLAGS.video,
        viz_img=flags.FLAGS.viz_img,
        output_path=flags.FLAGS.output_path,
    )


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))

ibc_logs/

data/block_push_states_location/
data/block_push_visual_location/
data/test_dataset/