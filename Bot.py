from evogym import get_full_connectivity
import gymnasium as gym
from helper import count_islands

class Bot:

    def __init__(self,body):
        self.body = body
        self.connections = get_full_connectivity(body)
        self.broken = not self.__verify_body(body)
        self.reward = 0
        if not self.broken:
            print(f'body:\n{self.body}\nconnections:{self.connections}')

    def __verify_body(self,body):
        return 1 == count_islands(body)

    def run(self):
        if self.broken:
            self.reward = -1
        else:
            try:
            
                env = gym.make(
                    'Walker-v0', 
                    body=self.body,
                    connections=self.connections, 
                    render_mode='human')
                env.reset()

                while True:
                    action = env.action_space.sample()
                    ob, reward, terminated, truncated, info = env.step(action)

                    if terminated or truncated:
                        self.reward = reward
                        env.close()
                        break
            except:
                self.reward = -1
