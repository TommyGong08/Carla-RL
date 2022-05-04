import glob
import os
import sys
import random
import time
import cv2
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm
import math
from collections import deque
from ResNet import *
from threading import Thread

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
SECOND_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 16
MINIBATCH_SIZE = 8
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
MODEL_NAME = "ResNet50"

MEMORY_FRACTION = 0.6
MIN_REWARD = -200

EPISODES = 100
DISCOUNT = 0.9
epsilon = 1
EPSILON_DECAY = 0.95  # 0.9975
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10
UPDATE_TARGET_EVERY = 5


# # Own Tensorboard class
# class ModifiedTensorBoard(TensorBoard):
#
#     # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.step = 1
#         self.writer = tf.summary.FileWriter(self.log_dir)
#
#     # Overriding this method to stop creating default log writer
#     def set_model(self, model):
#         pass
#
#     # Overrided, saves logs with our step number
#     # (otherwise every .fit() will start writing from 0th step)
#     def on_epoch_end(self, epoch, logs=None):
#         self.update_stats(**logs)
#
#     # Overrided
#     # We train for one batch only, no need to save anything at epoch end
#     def on_batch_end(self, batch, logs=None):
#         pass
#
#     # Overrided, so won't close writer
#     def on_train_end(self, _):
#         pass
#
#     # Custom method for saving own metrics
#     # Creates writer, writes custom metrics and closes writer
#     def update_stats(self, **stats):
#         self._write_logs(stats, self.step)


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_ANT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collison_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.front_camera = cv.resize(self.front_camera, dsize=(224, 224)) / 255
        return self.front_camera

    def collison_data(self, event):  # 碰撞相关的信息
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]  # 输出RGB三通道的值
        if self.SHOW_CAM:
            cv2.imshow("front camera rgb", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_ANT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_ANT))
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        if len(self.collision_hist) != 0:  # 有任何碰撞
            done = True
            reward = -200

        elif kmh < 50:
            done = False
            reward = -1

        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        self.front_camera = cv.resize(self.front_camera, dsize=(224, 224)) / 255
        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda")
        self.eval_model = self.create_model()
        self.target_model = self.create_model()
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

        # optimizer and loss
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()

    def create_model(self):
        model = ResNet_50(num_classes=3).to(self.device)
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.eval_model.state_dict())  # 更新目标网络权重
            self.target_update_counter = 0

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        for transition in minibatch:
            b_s = np.array(transition[0])
            b_s = F2.to_tensor(b_s).float()
            b_s = b_s.unsqueeze(0).cuda()
            b_a = transition[1]
            b_r = torch.tensor(transition[2]).cuda()
            b_s_ = np.array(transition[3])
            b_s_ = F2.to_tensor(b_s_).float()
            b_s_ = b_s_.unsqueeze(0).cuda()
            q_eval = self.eval_model(b_s)[:, b_a]
            q_next = self.target_model(b_s_).detach().max(1)[0]
            if not transition[4]:
                max_future_q = q_next
                q_target = b_r + DISCOUNT * max_future_q
            else:
                q_target = b_r
            q_target = q_target.float()
            loss = self.loss_func(q_eval, q_target)
            optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=0.01)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def get_qs(self, state):
        state = np.array(state)
        state = F2.to_tensor(state).float()
        state = state.unsqueeze(0).cuda()
        return self.eval_model(state).detach()

    # def training_in_loop(self):
    #     X = np.random.uniform(size=(480, 640, 3)).astype(np.float32)
    #     y = np.random.uniform(size=(1, 3)).astype(np.float32)
    #     x = cv.resize(X, dsize=(224, 224)) / 255
    #     x = F2.to_tensor(x)
    #     x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
    #     x = x.cuda()
    #     y = torch.tensor(y).to(self.device)
    #     y_ = self.eval_model(x)
    #     loss = self.loss_func(y_, y)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     self.training_initialized = True
    #
    #     while True:
    #         if self.terminate:
    #             print("[ERROR] Terminate!")
    #             return
    #         self.train()
    #         time.sleep(0.01)


if __name__ == '__main__':
    FPS = 20
    ep_rewards = [-200]
    random.seed(1)
    np.random.seed(1)

    if not os.path.isdir("models"):
        os.makedirs("models")

    agent = DQNAgent()
    env = CarEnv()

    # trainer_thread = Thread(target=agent.training_in_loop(), daemon=True)
    # trainer_thread.start()

    # while not agent.training_initialized:
    #     time.sleep(0.01)

    agent.get_qs(np.ones((244, 244, 3)))  # test
    torch.cuda.empty_cache()
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        env.collision_hist = []
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while True:
            if np.random.random() > epsilon:
                actions = agent.get_qs(current_state)
                action = torch.argmax(actions).item()
                # actions = action_tensor.numpy()
                # action = np.argmax(actions)
            else:
                action = np.random.randint(0, 3)
                time.sleep(1 / FPS)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state
            step += 1

            if len(agent.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
                agent.train()
                time.sleep(0.01)
                if done:
                    # print('Ep: ', episode, 'done')
                    break
            if done:
                break

        for actor in env.actor_list:
            actor.destroy()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
            #                                epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                torch.save(agent.target_model.state_dict(), f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}'
                                                            f'avg_{min_reward:_>7.2f}min__{int(time.time())}.pth')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    # trainer_thread.join()
    torch.save(agent.target_model.state_dict(), f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}'
                                                f'avg_{min_reward:_>7.2f}min__{int(time.time())}.pth')
