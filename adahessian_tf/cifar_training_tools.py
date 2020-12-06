import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm, tqdm_notebook
from adahessian import AdaHessian

def cifar_training(model, logdir, run_name, lr_values=[0.1, 0.01, 0.001], lr_boundaries=[16000, 24000, 32000],
                   val_interval=200, log_interval=200, batch_size=256, nesterov=False, optim_method='adahessian', weight_decay=0.):

    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundaries[:-1], values=lr_values)
    if optim_method == 'sgd':
        optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9, nesterov=nesterov)
    elif optim_method == 'adahessian':
        optimizer = AdaHessian(schedule, weight_decay= (weight_decay/lr_values[0]) )
    elif optim_method == 'adam':
        optimizer = tf.keras.optimizers.Adam(schedule)
    else:
        raise Exception(f'We do not support this {optim_method} yet. Please double check!')

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    ds = tfds.load('cifar10', as_supervised=True, data_dir='./')
    std = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
    mean= tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))

    def train_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.random_flip_left_right(x)
        x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - mean) / std
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = (x - mean) / std
        return x, y

    ds['train'] = ds['train'].map(train_prep).shuffle(10000).repeat().batch(batch_size).prefetch(-1)
    ds['test'] = ds['test'].map(valid_prep).batch(batch_size*4).prefetch(-1)

    runid = run_name + '_x' + str(np.random.randint(10000))
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    reg_loss = tf.metrics.Mean()

    print(f"RUNID: {runid}")
    tf.keras.utils.plot_model(model, os.path.join('saved_plots', runid + '.png'))

    @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            r_loss = tf.add_n(model.losses)
            outs = model(x, training)
            c_loss = loss_fn(y, outs)
            loss = c_loss + r_loss

        if training:
            
            if optim_method != 'adahessian':
                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            else:
                gradients, Hessian = optimizer.get_gradients_hessian(loss, model.trainable_weights)
                optimizer.apply_gradients_hessian(zip(gradients, Hessian, model.trainable_weights))


        accuracy(y, outs)
        cls_loss(c_loss)
        reg_loss(r_loss)


    training_step = 0
    best_validation_acc = 0
    epochs = lr_boundaries[-1] // val_interval

    for epoch in range(epochs):
        for x, y in tqdm(ds['train'].take(val_interval), desc=f'epoch {epoch+1}/{epochs}',
                         total=val_interval, ncols=100, ascii=True):

            training_step += 1
            step(x, y, training=True)

            if training_step % log_interval == 0:
                with writer.as_default():
                    c_loss, r_loss, err = cls_loss.result(), reg_loss.result(), 1-accuracy.result()
                    print(f" c_loss: {c_loss:^6.3f} | r_loss: {r_loss:^6.3f} | err: {err:^6.3f}", end='\r')

                    tf.summary.scalar('train/error_rate', err, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/regularization_loss', r_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    reg_loss.reset_states()
                    accuracy.reset_states()

        for x, y in ds['test']:
            step(x, y, training=False)
        print(f'Testing Acc at epoch {epoch}: {accuracy.result()}')
        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/error_rate', 1-accuracy.result(), step=training_step)

            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))

            cls_loss.reset_states()
            accuracy.reset_states()
        print(f'Best testing Acc at epoch {epoch}: {best_validation_acc}')