import numpy as np
from config import Config
from dsl import DSL
from karel.world import STATE_TABLE, World
from typing import Set, List, Tuple
from tasks.empty import EmptyTask

NORTH = 0
EAST = 1
SOUTH = 2
WEST  = 3

class WorldGenerator:

    def __init__(self) -> None:
        self.rng = np.random.RandomState(Config.env_seed)
        self.h = Config.env_height
        self.w = Config.env_width

    # this is the function that generates the worlds for the purpose of demonstrations. 
    def generate(self, wall_prob=0.1, marker_prob=0.1) -> World:
        s = np.zeros((self.h, self.w, len(STATE_TABLE)), dtype=bool)
        # Wall
        s[:, :, 4] = self.rng.rand(self.h, self.w) > 1 - wall_prob
        s[0, :, 4] = True
        s[self.h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, self.w-1, 4] = True
        # Karel initial location
        valid_loc = False
        while(not valid_loc):
            y = self.rng.randint(0, self.h)
            x = self.rng.randint(0, self.w)
            if not s[y, x, 4]:
                valid_loc = True
                s[y, x, self.rng.randint(0, 4)] = True
        # Marker: num of max marker == 1 for now TODO: this is the setting for LEAPS - do we keep it?
        s[:, :, 6] = (self.rng.rand(self.h, self.w) > 1 - marker_prob) * (s[:, :, 4] == False) > 0
        s[:, :, 5] = np.sum(s[:, :, 6:], axis=-1) == 0
        return World(s)
    

class EnvironmentGenerator:

    '''The following class receives a program and generates an Valid environment that the program solves.'''

    def __init__(self, dsl: DSL, agent_init_pos: Tuple[int] = (2,2), agent_init_direction: int = NORTH, seed: int = 0, p = 0.8,  max_num_exec: int = 150):
        self.dsl = dsl
        self._init_task = EmptyTask(seed)
        self.env = self._init_task.generate_state(agent_init_pos)
        self.agent_init_pos = agent_init_pos
        self.agent_init_direction = agent_init_direction
        self.rng = np.random.default_rng(seed)
        self.max_num_exec = max_num_exec
        self.env_height = Config.env_height
        self.env_width = Config.env_width
        self.p = p
        assert p >= 0 and p <= 1, "Probability Must be between 0 and 1."

    def reset_env(self):
        self.env = self._init_task.generate_state()

    def run_program(self, program_tokens: List[int]):
        program = self.dsl.parse_int_to_node(program_tokens)

        visited_hero_locations = set()
        self.final_pos = None

        num_exec = 0
        for _ in program.run_generator(self.env):
            visited_hero_locations.add(tuple(self.env.hero_pos))
            self.final_pos = self.env.hero_pos

            num_exec += 1
            if self.env.is_crashed() or num_exec > self.max_num_exec:
                    break
        xy_coords: Set = {(x,y) for x, y, _ in visited_hero_locations}
        
        return visited_hero_locations, xy_coords, self.final_pos

    def add_wall_to_state_here(self, coords: Tuple[int]):
        x, y = coords
        
        self.env.s[x, y, 4] = True

    def add_marker_here(self, coords: Tuple[int]):
        x, y = coords
        self.env.s[x, y, 6] = True

    def init_agent(self):
        
        # Set Last position to false:
        final_x, final_y, final_dir = self.final_pos 
        self.env.s[final_x, final_y, final_dir] = False
        # Set new position to true:
        x, y = self.agent_init_pos
        self.env.s[x, y, self.agent_init_direction] = True
        

    def generate_env(self, program_tokens: List[int]):
        visited_hero_locations, xy_coords, final_pos = self.run_program(program_tokens)

        xy_coords.add(self.agent_init_pos) 
        ### Generate walls. Choose random UNVISITED locations: 
        unvisited_xy = [(x,y) 
                  for x in range(self.env_height)
                    for y in range(self.env_width)
                        if (x,y) not in xy_coords
        ]
        
        
        # Now we choose a random subset of them to have walls. Do I choose a fix
        for coord in unvisited_xy:
            if self.rng.random() > self.p:
                self.add_wall_to_state_here(coord)
            
        # Put a Marker at the final position:
        final_x, final_y, _ = final_pos
        self.add_marker_here((final_x, final_y))

        # Set Karel to first state: 
        # TODO: Make this more dynamic:
        self.init_agent()

        return World(self.env.s) 
    
    def generate_dataset_of_envs(self):
        pass
