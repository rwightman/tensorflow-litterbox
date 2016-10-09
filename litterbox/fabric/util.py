
import tensorflow as tf
import os


def resolve_checkpoint_path(input_path):
    global_step = 0
    checkpoint_path = input_path
    if not os.path.exists(checkpoint_path):
        return '', global_step
    if not os.path.isfile(checkpoint_path):
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint_path = ckpt.model_checkpoint_path
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(input_path, checkpoint_path)
        else:
            return '', global_step

    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    try:
        global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
    except ValueError:
        pass

    return checkpoint_path, global_step


def check_tensorflow_version(min_version=11):
    assert int(str.split(tf.__version__,'.')[1]) >= min_version, \
        'Installed Tensorflow version (%s) is not be >= 0.%s.0' % (tf.__version__, min_version)
