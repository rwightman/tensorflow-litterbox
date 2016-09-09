# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import ast


default_Momentum = {'learning_rate': 0.1, 'momentum': 0.9, 'use_nesterov': True}
default_RMSProp = {'learning_rate': 0.1, 'decay': 0.9, 'momentum': 0.0, 'epsilon': 1e-10}
default_Adagrad = {'learning_rate': 0.1, 'initial_accumulator_value': 0.1}
default_Adadelta = {'learning_rate': 0.001, 'rho': 0.95, 'epsilon': 1e-8}
default_Adam = {'learning_rate': 0.001, 'beta1': 0.9, 'beta2':0.999, 'epsilon': 1e-8}

default_params = {
    'momentum': default_Momentum,
    'rmsprop': default_RMSProp,
    'adagrad': default_Adagrad,
    'adadelta': default_Adadelta,
    'adam': default_Adam
}

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('opt', 'default',
                           """Optimizer method. One of 'Momentum' (SGD), 'RMSProp', 'Adagrad', 'Adadelta', 'Adam'.
                           'Default' is specified to use the model's default optimizer.
                           """)

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation.

tf.app.flags.DEFINE_float('lr', None,
                          """Initial learning rate for training.
                          This specifies the rate at start of training
                          For constant, decay, or piecewise schedules.
                          Only the 'before decay' schedule will use a
                          different learning rate before this one.
                          This value overrides optimizer/model defaults
                          if values is not None.
                          """)

tf.app.flags.DEFINE_float('momentum', None,
                          """SGD momentum value.
                          This value overrides optimizer/model defaults
                          if values is not None.
                          """)

tf.app.flags.DEFINE_float('lr_epochs_per_decay', 25.0,
                          """Epochs after which learning rate decays. If
                          this is set to non zero, an exponential decay
                          learning rate schedule is used.
                          """)

tf.app.flags.DEFINE_float('lr_decay_factor', 0.2,
                          """Learning rate decay factor for exponential
                          decay schedule.""")

tf.app.flags.DEFINE_float('lr_epochs_priming', None,
                          """Number of epochs to use an initial 'priming'
                          learning rate before using the usual learning rate
                          schedule. If set to non-zero value, the priming
                          learning rate value will be used for that number of
                          epochs before handing off to normal rate schedule.
                          """)

tf.app.flags.DEFINE_float('lr_priming', 0.0001,
                          """The constant 'priming' learning rate that will be used
                          before the usual rate schedule. For networks that
                          are sensitive at start of training.
                          """)

tf.app.flags.DEFINE_string('lr_piecewise', None,
                           """A list of tuples in string literal format that define
                            the piecewise constant learning schedule.
                            For example, '[(0.5, 0.1), (10, 0.01], (20, 0.001)]'
                            This string equates to the following schedule:
                                Start training at specified or default initial learning rate.
                                After 0.5 epochs, train at a learning rate of 0.1.
                                After 10 epochs, train at a rate of 0.01.
                                After 20 epochs, train at a rate of 0.001.
                            """)


class OptParamScheduler(object):
    def __init__(self,
                 global_step_tensor,
                 num_steps_per_epoch=None,
                 model_default_type=None,
                 model_default_params=None):

        if not isinstance(global_step_tensor, tf.Variable):
            assert False, "Valid global_step tf variable/tensor is required"

        # properties
        self.num_steps_per_epoch = np.int64(num_steps_per_epoch)
        self.num_epochs_per_decay = FLAGS.lr_epochs_per_decay
        self.num_epochs_priming = FLAGS.lr_epochs_priming
        self.learning_rate_initial = FLAGS.lr
        self.learning_rate_priming = FLAGS.lr_priming
        self.learning_rate_decay_factor = FLAGS.lr_decay_factor
        self.learning_rate_piecewise_str = FLAGS.lr_piecewise

        self.opt_type = str.lower(FLAGS.opt)
        if self.opt_type == 'sgd':
            self.opt_type = 'momentum'

        # Prepare parameters from command line + defaults
        if (self.opt_type == 'default' and model_default_type not in default_params) or \
                self.opt_type not in default_params:
            print('Warning, invalid optimizer type specified, falling back to momentum.')
            self.opt_type = 'momentum'

        if self.opt_type == 'default':
            # If default params for model specified,
            # merge command line params appropriately
            self.opt_type = model_default_type  # 'momentum' if not specified
            self._opt_params = default_params[self.opt_type]
            if model_default_params:
                self._opt_params.update(model_default_params)
        else:
            self._opt_params = default_params[self.opt_type]

        if not self.learning_rate_initial:
            # Command line learning rate always takes priority.
            # If it's not specified, take rate from optimizer params.
            self.learning_rate_initial = self._opt_params['learning_rate']

        self.global_step_tensor = global_step_tensor
        self.learning_rate_tensor = None
        self.opt = None

    def initialize(self, override_learning_rate=None):

        if isinstance(override_learning_rate, tf.Variable):
            self.learning_rate_tensor = override_learning_rate  # override
        else:
            self._create_lr_schedule()

        # Override the learning rate passed to optimizer with tensor version
        self._opt_params['learning_rate'] = self.learning_rate_tensor

        self._create_optimizer()

        return self.opt

    def _create_optimizer(self):

        #FIXME add more overriding command line flags for different optimizers as needed
        if self.opt_type == 'momentum':
            if FLAGS.momentum:
                self._opt_params['momentum'] = FLAGS.momentum
            self.opt = tf.train.MomentumOptimizer(**self._opt_params)
        elif self.opt_type == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(**self._opt_params)
        elif self.opt_type == 'adagrad':
            self.opt = tf.train.AdagradOptimizer(**self._opt_params)
        elif self.opt_type == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(**self._opt_params)
        elif self.opt_type == 'adam':
            self.opt = tf.train.AdamOptimizer(**self._opt_params)
        else:
            assert False, 'Invalid optimizer %s specified' % self.opt_type

    def _create_lr_schedule(self):

        if self.learning_rate_piecewise_str:
            # Piecewise learning rate schedule
            # FIXME validate sooner, eg constructor?
            pieces = ast.literal_eval(self.learning_rate_piecewise_str)
            if pieces and len(pieces):
                boundaries, vals = zip(*pieces)
                boundaries = list(boundaries)
                values = [self.learning_rate_initial] + list(vals)
                learning_rate_tensor = self._create_lr_piecewise_sched(boundaries, values)
            else:
                assert False, "Invalid piecewise learning rate schedule %s" % FLAGS.lr_piecewise
        elif self.num_epochs_per_decay > 0:
            # Exponential learning rate decay schedule
            learning_rate_tensor = self._create_lr_exponential_sched(
                self.num_epochs_per_decay, self.learning_rate_decay_factor)
        else:
            # Constant learning rate schedule
            learning_rate_tensor = tf.convert_to_tensor(self.learning_rate_initial, name="learning_rate")

        if self.num_epochs_priming:
            # Priming phase added with an extra piecewise schedule
            # NOTE priming may stomp large portions of main schedule if not set correctly
            boundaries = [self.num_epochs_priming]
            values = [self.learning_rate_priming, learning_rate_tensor]
            learning_rate_tensor = self._create_lr_piecewise_sched(boundaries, values)

        self.learning_rate_tensor = learning_rate_tensor

    def _create_lr_piecewise_sched(self, boundaries, values):
        boundaries_in_steps = []
        for x in boundaries:
            boundaries_in_steps.append(np.int64(np.ceil(x * self.num_steps_per_epoch)))
        lr_tensor = tf.train.piecewise_constant(
            self.global_step_tensor,
            boundaries=boundaries_in_steps,
            values=values)
        return lr_tensor

    def _create_lr_exponential_sched(self, epochs_per_decay, decay_factor):
        decay_steps = int(self.num_steps_per_epoch * epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr_tensor = tf.train.exponential_decay(
            self.learning_rate_initial,
            self.global_step_tensor,
            decay_steps,
            decay_factor,
            staircase=True)
        return lr_tensor
