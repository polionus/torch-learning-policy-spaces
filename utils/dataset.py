from typing import List
from dsl import DSL 
from karel.world import World
from PIL import Image

def convert_world_to_image(world: World, save_path: str):
    frame = Image.fromarray(world.to_image())
    frame.convert('RGB').save(save_path)

def inspect_env(dsl:DSL, program_list: List, program_number: int, save_path_folder: str):

    _, prog, exec = program_list[program_number]
    states, _, _ = exec
    prog = dsl.parse_int_to_str(prog)
    for i in range(10):
        world_state = states[i, 1, :, :, :]
        world = World(world_state)
        save_path = f'{save_path_folder}/world-{i}.jpg'
        convert_world_to_image(world, save_path)
    print(f"Program: {prog}")
