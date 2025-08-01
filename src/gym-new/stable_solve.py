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

import gymnasium as gym
import network_sim
import tensorflow as tf

# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common.policies import FeedForwardPolicy
from stable_baselines3 import PPO
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common.simple_arg_parse import arg_or_default

arch_str = arg_or_default("--arch", default="32,16")
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch))

training_sess = None
config_file = arg_or_default("--config", default="/home/eecs/sarah/remy/configs/config0211.cfg")
name = arg_or_default("--name", default="test_run")
print(name)
print("Will save model to ./results/%s/" % name )
features = arg_or_default("--features", default="send rate,recv rate,latency ratio")
# class MyMlpPolicy(FeedForwardPolicy):

#     def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
#         super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=[{"pi":arch, "vf":arch}],
#                                         feature_extraction="mlp", **_kwargs)
#         global training_sess
#         training_sess = sess

env = gym.make('PccNs-v0', config=config_file, features=features)
# #env = gym.make('CartPole-v0')

gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO("MlpPolicy", env, verbose=1, gamma=gamma, batch_size=10000, n_steps=10000) #, n_steps=10000) # With MAX_STEPS=10000, the model will update 1 times per simulation in ~5 batches
# model = PPO.load("/home/eecs/sarah/PCC-RL/src/gym-new/results/cfg0211-noint-newtime-10k-10ksteps-10kb/pcc_model_ckpt0.zip", env) #TODO:Update the scrip tto optionally load from a checkpoint
# for i in range(0, 6):
#     with model.graph.as_default():                                                                   
#         saver = tf.train.Saver()                                                                     
#         saver.save(training_sess, "./pcc_model_%d.ckpt" % i)
# model.learn(total_timesteps=(1600 * 10000 * 6))
os.makedirs("./results/{}".format(name), exist_ok=True)
for i in range(0,6):
    model.learn(total_timesteps=(1600 * 10000))
    model.save("./results/{}/pcc_model_ckpt{}.zip".format(name, i))

# model.save("./results/{}/".format(name))
# ##
# #   Save the model to the location specified below.
# ##
# default_export_dir = "/tmp/pcc_saved_models/model_A/"
# export_dir = arg_or_default("--model-dir", default=default_export_dir)
# with model.graph.as_default():

#     pol = model.policy_pi#act_model

#     obs_ph = pol.obs_ph
#     act = pol.deterministic_action
#     sampled_act = pol.action

#     obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
#     outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
#     stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
#     signature = tf.saved_model.signature_def_utils.build_signature_def(
#         inputs={"ob":obs_input},
#         outputs={"act":outputs_tensor_info, "stochastic_act":stochastic_act_tensor_info},
#         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

#     #"""
#     signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                      signature}

#     model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
#     model_builder.add_meta_graph_and_variables(model.sess,
#         tags=[tf.saved_model.tag_constants.SERVING],
#         signature_def_map=signature_map,
#         clear_devices=True)
#     model_builder.save(as_text=True)
