import argparse
import functools
import os
import pathlib
import sys
from args_handler import define_config
from agent import Dreamer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

sys.path.append(str(pathlib.Path(__file__).parent))

import tools
import environments as envs


def main(config):
    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy('mixed_float16'))
    config.steps = int(config.steps)
    config.logdir.mkdir(parents=True, exist_ok=True)
    print('Logdir', config.logdir)

    # Create environments.
    datadir = config.logdir / 'episodes'
    writer = tf.summary.create_file_writer(
        str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    train_envs = [envs.wrappers.Async(lambda: envs.make_env(
        config, writer, 'train', datadir, store=True), config.parallel)
                  for _ in range(config.envs)]
    test_envs = [envs.wrappers.Async(lambda: envs.make_env(
        config, writer, 'test', datadir, store=False), config.parallel)
                 for _ in range(config.envs)]
    actspace = train_envs[0].action_space

    # Prefill dataset with random episodes.
    step = tools.count_steps(datadir, config)
    prefill = max(0, config.prefill - step)
    print(f'Prefill dataset with {prefill} steps.')
    random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
    tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
    writer.flush()

    # Train and regularly evaluate the agent.
    step = tools.count_steps(datadir, config)
    print(f'Simulating agent for {config.steps - step} steps.')
    agent = Dreamer(config, datadir, actspace, writer)
    if (config.logdir / 'variables.pkl').exists():
        print('Load checkpoint.')
        agent.load(config.logdir / 'variables.pkl')
    state = None
    while step < config.steps:
        print('Start evaluation.')
        tools.simulate(
            functools.partial(agent, training=False), test_envs, episodes=1)
        writer.flush()
        print('Start collection.')
        steps = config.eval_every // config.action_repeat
        state = tools.simulate(agent, train_envs, steps, state=state)
        step = tools.count_steps(datadir, config)
        agent.save(config.logdir / 'variables.pkl')
    for env in train_envs + test_envs:
        env.close()


if __name__ == '__main__':
    import colored_traceback

    colored_traceback.add_hook()
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
    main(parser.parse_args())
