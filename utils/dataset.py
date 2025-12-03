from typing import List
from dsl import DSL 
from karel.world import World
from PIL import Image


def inspect_env(dsl:DSL, program_list: List, program_number: int, save_path_folder: str):

    _, prog, exec = program_list[program_number]
    states, _, _ = exec
    prog = dsl.parse_int_to_str(prog)
    for i in range(10):
        world_state = states[i, 1, :, :, :]
        world = World(world_state)
        frame = Image.fromarray(world.to_image())
        frame.convert('RGB').save(f'{save_path_folder}/world-{i}.jpg')
    print('\n' * 4)
    print(prog)

