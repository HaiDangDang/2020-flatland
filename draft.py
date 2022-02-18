import functools

from abc import ABC, abstractmethod
import numpy as np
import os
from typing import Tuple
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin, PPOLoss, PPOTFPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.utils.explained_variance import explained_variance
from ray.tune import register_trainable
OTHER_AGENT = "other_agent"

class CentralizedCriticModel(ABC, TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.obs_space_shape = obs_space.shape[0]
        self.act_space_shape = action_space.n
        self.centralized = model_config["custom_options"]["critic"]["centralized"]
        self.max_num_agents = model_config["custom_options"]["max_num_agents"]
        self.max_num_opponents = self.max_num_agents - 1
        self.debug_mode = True

        # Build the actor network
        self.actor = self._build_actor(**model_config["custom_options"]["actor"])
        self.register_variables(self.actor.variables)

        # Build Central Value Network
        self.critic = self._build_critic(**model_config["custom_options"]["critic"])
        self.register_variables(self.critic.variables)

        if self.debug_mode:
            print("Actor_Model:\n", self.actor.summary())
            print("Critic_Model:\n", self.critic.summary())

    @abstractmethod
    def _build_actor(self, **kwargs) -> tf.keras.Model:
        pass

    @abstractmethod
    def _build_critic(self, **kwargs) -> tf.keras.Model:
        pass

    def forward(self, input_dict, state, seq_lens):
        policy = self.actor(input_dict["obs_flat"])
        self._value_out = tf.reduce_mean(input_tensor=policy, axis=-1)
        return policy, state

    def central_value_function(self, obs, other_agent):
        if self.centralized:
            return tf.reshape(self.critic([obs, other_agent]), [-1])
        return tf.reshape(self.critic(obs), [-1])

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class CcTransformer(CentralizedCriticModel):

    def _build_actor(self,activation_fn="relu", hidden_layers= [512, 512, 512], **kwargs):

        inputs = tf.keras.layers.Input(shape=(self.obs_space_shape,), name="obs")
        output = build_fullyConnected(
            inputs=inputs,
            hidden_layers=hidden_layers,
            num_outputs=self.act_space_shape,
            activation_fn=activation_fn,
            name="actor"
        )

        return tf.keras.Model(inputs, output)

    def _build_critic(self,
                      activation_fn="relu",
                      hidden_layers=[512, 512, 512],
                      centralized=True,
                      embedding_size=128,
                      num_head=8,
                      d_model=256,
                      use_scale=True,
                      **kwargs):
        agent_obs = tf.keras.layers.Input(shape=(self.act_space_shape,), name="obs")
        agent_embedding = build_fullyConnected(inputs=agent_obs,
                                               hidden_layers=[2*embedding_size,embedding_size],
                                               num_outputs=embedding_size,
                                               activation_fn=activation_fn,
                                               name="agent_embedding")

        opponent_shape = ((self.obs_space_shape + self.act_space_shape) * self.max_num_opponents, )
        opponent_obs = tf.keras.Input(shape=opponent_shape, name="other_agent")

        opponent_input = tf.reshape(
            opponent_obs,
            [-1, self.max_num_opponents, self.obs_space_shape + self.act_space_shape],
        )

        opponent_embedding = build_fullyConnected(
            inputs=opponent_input,
            hidden_layers=[2*embedding_size, embedding_size],
            num_outputs=embedding_size,
            activation_fn=activation_fn,
            name="opponent_embedding"
        )

        queries = tf.expand_dims(agent_embedding, axis=1)



def build_fullyConnected(inputs, hidden_layers, num_outputs, activation_fn="relu", name=None):

    name = name or "fc_network"

    x = inputs

    for k, layer_size in enumerate(hidden_layers):
        x = tf.keras.layers.Dense(
            layer_size,
            name="{}/fc_{}".format(name, k),
            activation=activation_fn,
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            bias_initializer=tf.keras.initializers.constant(0.1),
        )(x)

    output = tf.keras.layers.Dense(
        num_outputs,
        name="{}/fc_out".format(name),
        activation=None,
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        bias_initializer=tf.keras.initializers.constant(0.1),
    )(x)

    return output


class MultiHeadAttentionLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        use_scale: bool = True,
        use_residual_connection: bool = False,
        use_layer_norm: bool = True,
        **kwargs):

        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.use_scale = use_scale
        self.use_residual_connection = use_residual_connection
        self.use_layer_norm = use_layer_norm

        if d_model % self.num_heads != 0:
            raise ValueError(
                "the model dimension (got {}) must be a multiple "
                "of the number of heads, got {} heads".format(d_model, num_heads)
            )

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=True)

        self.attention_layer = AttentionLayer(use_scale=self.use_scale)
        self.transition_layer = tf.keras.layers.Dense(d_model)

        if self.use_layer_norm:
            self.layer_norm = tf.keras.layers.LayerNormalization

    def _split_head(self, inputs):

        inputs = tf.concat(tf.split(inputs, self.num_heads, axis=-1), axis=0)
        return inputs






class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, use_scale=True, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.use_scale = use_scale

    def build(self, input_shape):
        if self.use_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=(),
                initializer=tf.constant_initializer(1.0),
                trainable=True,
            )
        else:
            self.scale = 1.0 / tf.sqrt(tf.cast(input_shape[0][-1]), tf.float32)

    def call(self, inputs):

        self._validate_call_arg(inputs=inputs)
        q,k,v = inputs[0], inputs[1], inputs[2]

        scores = self._calculate_scores(query=q, key=k)
        result = self._apply_scores(scores=scores, value=v)
        return result

    def _calculate_scores(self, query, key):

        return  self.scale * tf.matmul(query, key, transpose_b=True, name="scores")

    def _apply_scores(self, scores, value):
        attention_weights = tf.nn.softmax(scores, axis=-1, name="attention_weights")
        output = tf.matmul(attention_weights, value)
        return output

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config["use_scale"] = self.use_scale
        return  config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _validate_call_arg(self, inputs):
        class_name = self.__class__.__name__
        if not isinstance(inputs, list):
            raise ValueError(
                "{} layer must be called on a list of inputs, namely "
                "[query, value, key].".format(class_name)
            )
        if len(inputs) != 3:
            raise ValueError(
                "{} layer accepts inputs list of length 3, "
                "namely [query, value, key]. "
                "Given length: {}".format(class_name, len(inputs))
            )