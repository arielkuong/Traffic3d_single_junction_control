# from abc import ABC, abstractmethod
import socket
import cv2
import os
import tempfile
import json
import numpy as np
from datetime import datetime

class Traffic3DSingleJunction:
    PORT = 13000
    def __init__(self, port=PORT):
        self.port = port
        self.max_number_of_junction_states = 0
        self.client_socket = None
        self.max_timesteps = 100
        self.env_timeout = 0

        # Temp variables, deleted when env reset is_available
        self.original_start = False
        self.last_obs = []

        # setup the location to store screenshot images
        # self.images_path = os.path.join(tempfile.gettempdir(), "Traffic3D_Screenshots", datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f"))
        self.images_path = os.path.join(tempfile.gettempdir(), "Traffic3D_Screenshots", datetime.now().strftime("%Y-%m-%d_%H"))
        os.makedirs(self.images_path, exist_ok=True)
        print("Screenshots are located at: " + self.images_path)

        self._setup_socket()

    def _get_data(self):
        return self.client_socket.recv(1024)

    def _send_data(self, data_to_send):
        action_bytes = bytes(str(data_to_send), "ascii")
        self.client_socket.send(action_bytes)

    def _setup_socket(self):
        ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ss.bind(("0.0.0.0", self.port))
        ss.listen()
        print("waiting for tcpConnection")
        (self.client_socket, address) = ss.accept()
        print("tcpConnection established")
        self._send_data(self.images_path)
        # self.max_number_of_junction_states = 4
        self.max_number_of_junction_states = int(self._get_data().decode('utf-8'))
        if self.max_number_of_junction_states == 0:
            raise ValueError("The Max Number of Junction States is 0. It is possible that Traffic3D "
                             "never sent the number in the first place or there are no Junction States in Traffic3D.")
        print("Max Junction State Size: " + str(self.max_number_of_junction_states))
        self.env_timeout = 0

    def _receive_image(self):
        data_string = (self._get_data().decode('utf-8'))
        # print(data_string)
        screenshots = json.loads(data_string)
        screenshots = screenshots["screenshots"]
        screenshots = screenshots[0]
        img_path = os.path.join(self.images_path, screenshots["screenshotPath"])
        # img_path = os.path.join(self.images_path, data_string)
        # print('Receiving image...')
        img = cv2.imread(img_path)
        # print('image received from: {}'.format(img_path))
        return img

    def _send_action(self, action_input):
        # print(action)
        actions = {"actions": []}
        action = np.array(action_input)
        action = action.tolist()
        actions["actions"].append({"junctionId": '1', "action": action})
        self._send_data(json.dumps(actions))
        # self._send_data(action)
        # print("action sent")

    def _receive_reward(self):
        data = (self._get_data().decode('utf-8'))
        data_float = float(data)
        # print('reward received: {:.2f}'.format(data_float))
        return data_float

    def get_action_space(self):
        return self.max_number_of_junction_states

    def reset(self):
        if self.original_start == False:
        # self._setup_socket()
            # print('Traffic3DSingleJunction: Original reset, receive the first image')
            ob = self._receive_image()
            self.original_start = True
        else:
            ob = self.last_obs
            # print('Traffic3DSingleJunction: Not origianl reset, just copy last obs')
        self.env_timeout = 0
        return ob

    def step(self, action):
        self._send_action(action)
        reward = self._receive_reward()
        obs = self._receive_image()
        done = False
        info = {}
        self.env_timeout += 1
        if self.env_timeout >= self.max_timesteps:
            done = True
            self.last_obs = obs

        return obs, reward, done, info




    # def render(self)
