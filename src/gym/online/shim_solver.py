import gym
import shim_env
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO1
from stable_baselines import TRPO
from simple_arg_parse import arg_or_default

arch_str = arg_or_default("--arch", default="32,16")
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch))

my_sess = None

class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=[{"pi":arch, "vf":arch}],
                                        feature_extraction="mlp", **_kwargs)
        global my_sess
        my_sess = sess

n_cpu = 1
env = gym.make('NetShim-v0')
#env = gym.make('CartPole-v0')

gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)

"""
with model.graph.as_default():
    saver = tf.train.Saver()
    saver.restore(my_sess, "/home/njay2/tmp_saved_model.ckpt")
"""

with model.graph.as_default():
    print(tf.global_variables())
    var_23 = [v for v in tf.global_variables() if v.name == "model/vf/w:0"][0]
    print(my_sess.run(var_23))
    saver = tf.train.Saver()
    saver.restore(my_sess, "/home/njay2/tmp_saved_model.ckpt")
    var_23 = [v for v in tf.global_variables() if v.name == "model/vf/w:0"][0]
    print(my_sess.run(var_23))

model.learn(total_timesteps=(9600 * 410))

##
#   Save the model to the location specified below.
##
default_export_dir = "/tmp/pcc_saved_models/model_A/"
export_dir = arg_or_default("--model-dir", default=default_export_dir)
with model.graph.as_default():

    pol = model.policy_pi#act_model

    obs_ph = pol.obs_ph
    act = pol.deterministic_action
    sampled_act = pol.action

    obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
    outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
    stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"ob":obs_input},
        outputs={"act":outputs_tensor_info, "stochastic_act":stochastic_act_tensor_info},
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
