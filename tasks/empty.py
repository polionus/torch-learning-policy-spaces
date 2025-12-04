import numpy as np
from karel.world import World, STATE_TABLE 
from typing import Tuple
from .task import Task

### In order to create the state, the last dimension of the state is the type . 

class EmptyTask(Task):
    
    #This function is called once in the task.py file.
    def generate_state(self, agent_init_pos: Tuple[int] = (2,2), agent_init_direction: int = 0):
        state = np.zeros(
            (self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool
        )
        agent_x, agent_y = agent_init_pos
        state[agent_x, agent_y, agent_init_direction] = True
        return World(state)

    def reset_state(self) -> None:
        world_state = super().reset_state()
        return world_state

    ### This function is called when running the program or policy in the env.
    def get_reward(self, world_state: World):

        terminated = False
        reward = 0

        return terminated, reward
