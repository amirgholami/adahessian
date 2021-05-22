

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


import abc
import contextlib
import functools

import six
import tensorflow as tf
import numpy as np
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export

class AdaHessian(optimizer_v2.OptimizerV2):

    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
               learning_rate=0.1,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-4,
               weight_decay = 0.,
               hessian_power=1.0,
               name='AdaHessian',
               average_size_1d=None,
               average_size_2d=None,
               average_size_3d=-1,
               average_size_4d=-1,
               **kwargs):
        """Construct a new AdaHessian optimizer.
        Args:
            learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
            takes no arguments and returns the actual value to use, The learning
            rate. Defaults to 0.1.
            beta_1: A float value or a constant float tensor, or a callable that takes
            no arguments and returns the actual value to use. The exponential decay
            rate for the 1st moment estimates. Defaults to 0.9.
            beta_2: A float value or a constant float tensor, or a callable that takes
            no arguments and returns the actual value to use, The exponential decay
            rate for the 2nd moment estimates. Defaults to 0.999.
            epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
            1e-7.
            weight_decay: We are using AdamW's weight decay scheme. Defaults to 0.
            name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".
            hessian_power: Hessian power to control the optimizer more similar to first/second 
            order method (default: 1). You can also try 0.5. For some tasks we found this 
            to result in better performance.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
            # average_size_{1,2,3,4}d: 
                        None: use no spatial averaging
                        -1: use suggested spatial averaging (recommended for conv kernels)
                        >= 1: use customized size
        """

        super(AdaHessian, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.weight_decay = weight_decay
        self.hessian_power = hessian_power
        self.average_size_1d = average_size_1d
        self.average_size_2d = average_size_2d
        self.average_size_3d = average_size_3d
        self.average_size_4d = average_size_4d

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdaHessian, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = (
            apply_state[(var_device, var_dtype)]['lr_t'] *
            (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdaHessian, self).set_weights(weights)


    def get_gradients_hessian(self, loss, params):
        """Returns gradients and Hessian of `loss` with respect to `params`.
        Arguments:
            loss: Loss tensor.
            params: List of variables.
        Returns:
            List of gradient and Hessian tensors.
        Raises:
            ValueError: In case any gradient cannot be computed (e.g. if gradient
            function not implemented).
        """
        params = nest.flatten(params)
        with backend.get_graph().as_default(), backend.name_scope(self._name +
                                                                    "/gradients"):
            grads = gradients.gradients(loss, params)
            for grad, param in zip(grads, params):
                if grad is None:
                    raise ValueError("Variable {} has `None` for gradient. "
                                    "Please make sure that all of your ops have a "
                                    "gradient defined (i.e. are differentiable). "
                                    "Common ops without gradient: "
                                    "K.argmax, K.round, K.eval.".format(param))

            # WARNING: for now we do not support gradient clip
            # grads = self._clip_gradients(grads)

            v = [np.random.uniform(0, 1, size = p.shape) for p in params]
            for vi in v:
                vi[ vi < 0.5] =  -1 
                vi[ vi >= 0.5] =  1 
            v = [tf.convert_to_tensor(vi, dtype = tf.dtypes.float32) for vi in v]

            vprod = tf.reduce_sum([ tf.reduce_sum(vi * grad) for vi, grad in zip(v, grads)])

            Hv = gradients.gradients(vprod, params)

            Hd = [ tf.abs(Hvi * vi) for Hvi, vi in zip(Hv, v)]

        return grads, Hd

    def _filter_grads_hessian(self, grads_hessian_and_vars):
        """Filter out iterable with grad equal to None."""
        grads_hessian_and_vars = tuple(grads_hessian_and_vars)
        if not grads_hessian_and_vars:
            return grads_hessian_and_vars
        filtered = []
        vars_with_empty_grads = []
        for grad, hessian, var in grads_hessian_and_vars:
            if grad is None:
                vars_with_empty_grads.append(var)
            else:
                filtered.append((grad, hessian, var))
        filtered = tuple(filtered)

        if not filtered:
            raise ValueError("No gradients provided for any variable: %s." %
                            ([v.name for _, v in grads_and_vars],))
        if vars_with_empty_grads:
            logging.warning(
                ("Gradients do not exist for variables %s when minimizing the loss."),
                ([v.name for v in vars_with_empty_grads]))
        return filtered

    def apply_gradients_hessian(self,
                      grads_hessian_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
        grads_hessian_and_vars = self._filter_grads_hessian(grads_hessian_and_vars)
        var_list = [v for (_, _, v) in grads_hessian_and_vars]

        with backend.name_scope(self._name):
            # Create iteration if necessary.
            with ops.init_scope():
                self._create_all_weights(var_list)

        if not grads_hessian_and_vars:
            # Distribution strategy does not support reducing an empty list of
            # gradients
            return control_flow_ops.no_op()

        if distribute_ctx.in_cross_replica_context():
            raise RuntimeError(
                "`apply_gradients() cannot be called in cross-replica context. "
                "Use `tf.distribute.Strategy.run` to enter replica "
                "context.")

        strategy = distribute_ctx.get_strategy()
        if (not experimental_aggregate_gradients and strategy and isinstance(
            strategy.extended,
            parameter_server_strategy.ParameterServerStrategyExtended)):
            raise NotImplementedError(
                "`experimental_aggregate_gradients=False is not supported for "
                "ParameterServerStrategy and CentralStorageStrategy")

        apply_state = self._prepare(var_list)
        if experimental_aggregate_gradients:
            reduced_grads, reduced_hessian = self._aggregate_gradients_hessian(grads_hessian_and_vars)
            var_list = [v for _, _, v in grads_hessian_and_vars]
            grads_hessian_and_vars = list(zip(reduced_grads, reduced_hessian, var_list))


        return distribute_ctx.get_replica_context().merge_call(
            functools.partial(self._distributed_apply, apply_state=apply_state),
            args=(grads_hessian_and_vars,),
            kwargs={
                "name": name,
            })

    def _aggregate_gradients_hessian(self, grads_hessian_and_vars):
        """Returns all-reduced gradients.
        Args:
        grads_and_vars: List of (gradient, hessian, variable) pairs.
        Returns:
        Two lists of all-reduced gradients and Hessian.
        """
        grads_hessian_and_vars = list(grads_hessian_and_vars)
        filtered_grads_hessian_and_vars = self._filter_grads_hessian(grads_hessian_and_vars)

        # split the list so that we can use the all_recude_fn
        filtered_grads_and_vars = tuple([(g, v) for (g, h, v) in filtered_grads_hessian_and_vars])
        filtered_hessian_and_vars = tuple([(h, v) for (g, h, v) in filtered_grads_hessian_and_vars])


        def all_reduce_fn(distribution, grads_hessian_and_vars):
            # WARNING: this ReduceOp.SUM can only support two entries, for now we have three.
            # So far now, we do it for two steps to make life easier.
            return distribution.extended.batch_reduce_to(
                ds_reduce_util.ReduceOp.SUM, grads_hessian_and_vars)

        if filtered_grads_hessian_and_vars:
            reduced_part1 = distribute_ctx.get_replica_context().merge_call(
                all_reduce_fn, args=(filtered_grads_and_vars,))
            reduced_part2 = distribute_ctx.get_replica_context().merge_call(
                all_reduce_fn, args=(filtered_hessian_and_vars,))
        else:
            reduced = []

        # Copy 'reduced' but add None gradients back in
        reduced_with_nones_grads = []
        reduced_with_nones_hessian = []

        reduced_pos = 0
        for g, h, _ in grads_hessian_and_vars:
            if g is None:
                reduced_with_nones_grads.append( None )
                reduced_with_nones_hessian.append( None )
            else:
                reduced_with_nones_grads.append(reduced_part1[reduced_pos])
                reduced_with_nones_hessian.append(reduced_part2[reduced_pos])
                reduced_pos += 1

        return reduced_with_nones_grads, reduced_with_nones_hessian


    @def_function.function(experimental_compile=True)
    def _resource_apply_dense(self, grad, hess, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                        self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        m.assign_add((grad - m) * (1 - coefficients['beta_1_t']))
        # this part need to be changed for spatial averaging

        if len(v.shape) == 1:
            resize = self.average_size_1d
        elif len(v.shape) == 2:
            resize = self.average_size_2d
        elif len(v.shape) == 3:
            resize = self.average_size_3d
        elif len(v.shape) == 4:
            resize = self.average_size_4d
        else:
            raise Exception('You need to define the spatial average size by yourself!')

        if resize == None:
            v.assign_add((math_ops.square(hess) - v) * (1 - coefficients['beta_2_t']))
        elif resize == -1:
            if len(v.shape) == 1:
                v.assign_add((math_ops.square(hess) - v) * (1 - coefficients['beta_2_t']))
            elif len(v.shape) == 2:
                hess_average = tf.reduce_mean(hess, [0], keepdims=True)
                v.assign_add((math_ops.square(hess_average) - v) * (1 - coefficients['beta_2_t'])) 
            elif len(v.shape) == 3:
                hess_average = tf.reduce_mean(hess, [0], keepdims=True)
                v.assign_add((math_ops.square(hess_average) - v) * (1 - coefficients['beta_2_t'])) 
            elif len(v.shape) == 4:
                hess_average = tf.reduce_mean(hess, [0, 1], keepdims=True)
                v.assign_add((math_ops.square(hess_average) - v) * (1 - coefficients['beta_2_t'])) 
        else:
            if resize <= 0:
                raise Exception('You need to define the spatial average size >= 1!')
            hess_average = tf.reshape(hess, [resize, -1])
            hess_average = tf.reduce_mean(hess_average, [0])
            hess_average = tf.repeat(hess_average, resize)
            hess_average = tf.reshape(hess_average, v.shape)
            v.assign_add((math_ops.square(hess_average) - v) * (1 - coefficients['beta_2_t']))
        
        bias_correct1 = 1 - coefficients['beta_1_power']
        bias_correct2 = 1 - coefficients['beta_2_power']

        if self.weight_decay != 0:
            var.assign_sub(coefficients['lr_t'] * self.weight_decay * var)

        # denom = np.power(math_ops.sqrt(v / bias_correct2), self.hessian_power) + coefficients['epsilon']
        denom = tf.math.pow(math_ops.sqrt(v / bias_correct2), self.hessian_power) + coefficients['epsilon']


        var.assign_sub( coefficients['lr_t'] * m / bias_correct1 / denom  )

    @def_function.function(experimental_compile=True)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise Exception('For now, we do not support sparse update yet.')

    def get_config(self):
        config = super(AdaHessian, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay
        })
        return config

    def _distributed_apply(self, distribution, grads_hessian_and_vars, name, apply_state):
        """`apply_gradients` using a `DistributionStrategy`."""

        def apply_grad_to_update_var(var, grad, hess):
            """Apply gradient to variable."""
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)

            apply_kwargs = {}

            if "apply_state" in self._dense_apply_args:
                apply_kwargs["apply_state"] = apply_state
            update_op = self._resource_apply_dense(grad, hess, var, **apply_kwargs)

            if var.constraint is not None:
                with ops.control_dependencies([update_op]):
                    return var.assign(var.constraint(var))
            else:
                return update_op

        eagerly_outside_functions = ops.executing_eagerly_outside_functions()
        update_ops = []
        with ops.name_scope(name or self._name, skip_on_eager=True):
            for grad, hess, var in grads_hessian_and_vars:

                def _assume_mirrored(grad, hess):
                    if isinstance(grad, ds_values.PerReplica):
                        return ds_values.Mirrored(grad.values), ds_values.Mirrored(hess.values)
                    return grad, hess

                grad, hess = nest.map_structure(_assume_mirrored, grad, hess)
                # Colocate the update with variables to avoid unnecessary communication
                # delays. See b/136304694.
                with distribution.extended.colocate_vars_with(var):
                    with ops.name_scope("update" if eagerly_outside_functions else
                                  "update_" + var.op.name, skip_on_eager=True):
                        update_ops.extend(distribution.extended.update(
                                var, apply_grad_to_update_var, args=(grad, hess), group=False))

            any_symbolic = any(isinstance(i, ops.Operation) or
                             tf_utils.is_symbolic_tensor(i) for i in update_ops)
            if not context.executing_eagerly() or any_symbolic:
                # If the current context is graph mode or any of the update ops are
                # symbolic then the step update should be carried out under a graph
                # context. (eager updates execute immediately)
                with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
                    with ops.control_dependencies(update_ops):
                        return self._iterations.assign_add(1, read_value=False)

            return self._iterations.assign_add(1)
