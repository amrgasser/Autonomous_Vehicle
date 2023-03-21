from utils.rewards_utils import discount_rewards
from utils.crashing_utils import check_crash, check_exceed_max_rot, check_out_of_lane
from utils.training_rewards_utils import compute_driving_loss, train_step
from Models.CNN import CNN
from classes.LossHistory import LossHistory
from classes.PeriodicPlotter import PeriodicPlotter
from classes.Memory import Memory
import tensorflow as tf
import gym
import time
import io
import base64
import tensorflow_probability as tfp
import functools
import IPython
import os
from tqdm import tqdm
import numpy as np
from utils.vista_utils import vista_step
from utils.utils import grab_and_preprocess_obs
import vista
from vista.utils import logging
logging.setLevel(logging.ERROR)


trace_root = "./vista_traces"
trace_path = [
    "20210726-154641_lexus_devens_center",
    "20210726-155941_lexus_devens_center_reverse",
    "20210726-184624_lexus_devens_center",
    "20210726-184956_lexus_devens_center_reverse",
]
trace_path = [os.path.join(trace_root, p) for p in trace_path]

# Create a virtual world with VISTA, the world is defined by a series of data traces
world = vista.World(trace_path, trace_config={'road_width': 4})

# Create a car in our virtual world. The car will be able to step and take different
#   control actions. As the car moves, its sensors will simulate any changes it environment
car = world.spawn_agent(
    config={
        'length': 5.,
        'width': 2.,
        'wheel_base': 2.78,
        'steering_ratio': 14.7,
        'lookahead_road': True
    })

camera = car.spawn_camera(config={'size': (200, 320)})
display = vista.Display(world, display_config={
    "gui_scale": 2, "vis_full_frame": False})

memory = Memory()
model = CNN()
driving_model = model.driving_model


def vista_reset():
    world.reset()
    display.reset()


def train_model():

    learning_rate = 4e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # instantiate driving agent
    vista_reset()

    # compute_driving_loss = function(model.)
    # compute_driving_loss = function(model.run_driving_model)

    # to track our progress
    smoothed_reward = LossHistory(smoothing_factor=0.9)
    plotter = PeriodicPlotter(
        sec=2, xlabel='Iterations', ylabel='Rewards')
    memory = Memory()

    max_batch_size = 300
    max_reward = float('-inf')
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()  # clear if it exists
    for i_episode in range(500):
        plotter.plot(smoothed_reward.get())
        # Restart the environment
        vista_reset()
        memory.clear()
        observation = grab_and_preprocess_obs(car, camera)

        while True:
            curvature_dist = model.run_driving_model(observation)
            curvature_action = curvature_dist.sample()[0, 0]

            # Step the simulated car with the same action
            vista_step(curvature_action, car=car)
            observation = grab_and_preprocess_obs(car, camera)

            reward = 1.0 if not check_crash(car) else 0.0

            memory.add_to_memory(observation, curvature_action, reward)

            if reward == 0.0:
                total_reward = sum(memory.rewards)
                smoothed_reward.append(total_reward)
                batch_size = min(len(memory), max_batch_size)
                i = np.random.choice(len(memory), batch_size, replace=False)
                train_step(driving_model, compute_driving_loss, optimizer,
                           observations=np.array(memory.observations)[i],
                           actions=np.array(memory.actions)[i],
                           discounted_rewards=discount_rewards(
                               memory.rewards)[i],
                           custom_fwd_fn=model.run_driving_model)
                # reset the memory
                memory.clear()
                break


def evaluate_model():
    i_step = 0
    num_episodes = 5
    num_reset = 5
    stream = VideoStream()
    for i_episode in range(num_episodes):

        # Restart the environment
        vista_reset()
        observation = grab_and_preprocess_obs(car)

        print("rolling out in env")
        episode_step = 0
        while not check_crash(car) and episode_step < 100:
            # using our observation, choose an action and take it in the environment
            curvature_dist = model.run_driving_model(observation)
            curvature = curvature_dist.mean()[0, 0]

            # Step the simulated car with the same action
            vista_step(curvature)
            observation = grab_and_preprocess_obs(car)

            vis_img = display.render()
            stream.write(vis_img[:, :, ::-1], index=i_step)
            i_step += 1
            episode_step += 1

        for _ in range(num_reset):
            stream.write(np.zeros_like(vis_img), index=i_step)
            i_step += 1

    print(
        f"Average reward: {(i_step - (num_reset*num_episodes)) / num_episodes}")

    print("Saving trajectory with trained policy...")
    stream.save("./trained_policy.mp4")


train_model()
