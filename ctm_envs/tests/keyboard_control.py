from pynput.keyboard import Key, Listener
import gym
import time
import numpy as np
import argparse

import ctm_envs


class KeyboardControl(object):
    def __init__(self, env_id, env_kwargs):
        self.env = gym.make(env_id, **env_kwargs)

        self.key_listener = Listener(on_press=self.on_press_callback)
        self.key_listener.start()

        self.action = np.zeros_like(self.env.action_space.low)
        self.extension_actions = np.zeros(3)
        self.rotation_actions = np.zeros(3)

        self.extension_value = self.env.action_space.high[0] / 2
        self.rotation_value = self.env.action_space.high[-1] / 2
        self.exit = False

    def on_press_callback(self, key):
        # Tube 1 (inner most tube) is w s a d
        # Tube 2 (outer most tube) is t g f h
        # Tube 3 (outer most tube) is i k j l
        try:
            if key.char in ['w', 's', 'a', 'd']:
                if key.char == 'w':
                    self.extension_actions[0] = self.extension_value
                elif key.char == 's':
                    self.extension_actions[0] = -self.extension_value
                elif key.char == 'a':
                    self.rotation_actions[0] = self.rotation_value
                elif key.char == 'd':
                    self.rotation_actions[0] = -self.rotation_value
            if key.char in ['t', 'g', 'f', 'h']:
                if key.char == 't':
                    self.extension_actions[1] = self.extension_value
                elif key.char == 'g':
                    self.extension_actions[1] = -self.extension_value
                elif key.char == 'f':
                    self.rotation_actions[1] = self.rotation_value
                elif key.char == 'h':
                    self.rotation_actions[1] = -self.rotation_value
            if key.char in ['i', 'k', 'j', 'l']:
                if key.char == 'i':
                    self.extension_actions[2] = self.extension_value
                elif key.char == 'k':
                    self.extension_actions[2] = -self.extension_value
                elif key.char == 'j':
                    self.rotation_actions[2] = self.rotation_value
                elif key.char == 'l':
                    self.rotation_actions[2] = -self.rotation_value
        except AttributeError:
            if key == Key.esc:
                self.exit = True
                exit()
            else:
                self.extension_actions = np.zeros(3)
                self.rotation_actions = np.zeros(3)

    def run(self):
        obs = self.env.reset()
        while not self.exit:
            self.action[:3] = self.extension_actions
            self.action[3:] = self.rotation_actions
            # print('action: ', self.action)
            observation, reward, done, info = self.env.step(self.action)
            self.extension_actions = np.zeros(3)
            self.rotation_actions = np.zeros(3)
            self.env.render()
            if info['is_success']:
                obs = self.env.reset()
            self.action = np.zeros_like(self.env.action_space.low)
            time.sleep(0.1)
        self.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="CTR-Reach-v0")
    parser.add_argument('--model', type=str, default="exact")
    parser.add_argument('--joint-representation', type=str, default="trig")
    parser.add_argument('--render', type=bool, default=True)
    args = parser.parse_args()

    env_kwargs = {'model': args.model, 'joint_representation': args.joint_representation, 'render': args.render}

    keyboard_agent = KeyboardControl(args.env, env_kwargs)
    keyboard_agent.run()
