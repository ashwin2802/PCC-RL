# Copyright 2019 Nathan Jay and Noga Rotman
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

import gym
import network_sim
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy
# from stable_baselines.td3.policies import FeedForwardPolicy
# from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines import PPO1, TD3, SAC
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common.simple_arg_parse import arg_or_default

arch_str = arg_or_default("--arch", default="64,32")
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch))

training_sess = None

class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=[{"pi":arch, "vf":arch}],
                                        feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess
    # def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
    #     super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env=n_env, n_steps=n_steps, n_batch=n_batch, reuse=reuse, layers=arch,
    #                                         feature_extraction="mlp", **_kwargs)
    #     global training_sess
    #     training_sess = sess

# class MyCnnPolicy(FeedForwardPolicy):
#     def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
#         super(MyCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
#                                         feature_extraction="cnn", **_kwargs)
#         global training_sess
#         training_sess =sess

env = gym.make('PccNs-v0')
#env = gym.make('CartPole-v0')

gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO1(MyMlpPolicy, env, verbose=1, tensorboard_log='./ppo_tb_log/lowl', schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)
# model = TD3(MyMlpPolicy, env, learning_rate=0.0003, gamma=gamma, verbose=1, tensorboard_log='./td3_tb_log/arch2', action_noise=NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=float(0.5)*np.ones(env.action_space.shape[-1])))
# model = SAC(MyMlpPolicy, env, learning_rate=0.0003, gamma=gamma, verbose=1, tensorboard_log='./sac_tb_log/arch2', action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=float(0.5)*np.ones(env.action_space.shape[-1])))

for i in range(0, 6):
    with model.graph.as_default():                                                                   
        saver = tf.train.Saver()                                                                     
        saver.save(training_sess, "./runs/ppo_hight/pcc_model_%d.ckpt" % i)
    model.learn(total_timesteps=(1600*410))

with model.graph.as_default():                                                                   
    saver = tf.train.Saver()                                                                     
    saver.restore(training_sess, "./runs/ppo_hight/pcc_model_4.ckpt")

##
#   Save the model to the location specified below.
##
default_export_dir = "/tmp/pcc_saved_models/model_ppo_hight/"
export_dir = arg_or_default("--model-dir", default=default_export_dir)
with model.graph.as_default():

    pol = model.policy_pi 
    #act_model
    # pol = model.policy_tf

    obs_ph = pol.obs_ph
    act = pol.deterministic_action
    #TD3
    # act = unscale_action(model.action_space, model.policy_out) 
    # act = unscale_action(model.action_space, model.deterministic_action)
    sampled_act = pol.action

    obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
    outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
    stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"ob":obs_input},
        outputs={"act":outputs_tensor_info, "stochastic_act":stochastic_act_tensor_info},
        # outputs={"act":outputs_tensor_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    #"""
    signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                     signature}

    model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    model_builder.add_meta_graph_and_variables(model.sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_map,
        clear_devices=True)
    model_builder.save(as_text=True)


print("hight")