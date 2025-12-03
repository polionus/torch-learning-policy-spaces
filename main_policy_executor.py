import torch
# from dsl.parser import Parser
from dsl import DSL
from typing import List
from vae.models.leaps_vaeLSTM import LeapsVAELSTM
from vae.models.leaps_vaeAttention import LeapsVAEAttention
from vae.models.leaps_vae import LeapsVAE
from functools import partial
from tasks.empty import EmptyTask
import numpy as np
from vae.program_dataset import load_programs
import torch.nn as nn

PROG = 1
PROGRAMS = [
    'DEF run m( IFELSE c( frontIsClear c) i( move i) ELSE e( turnRight e) m)',
    'DEF run m( REPEAT R=4 r( move turnRight r) m)',
    'DEF run m( WHILE c( frontIsClear c) w( move w) m)',
    'DEF run m( WHILE c( rightIsClear c) w( turnRight IFELSE c( markersPresent c) i( pickMarker i) ELSE e( move e) w) m)',
    'DEF run m( IFELSE c( noMarkersPresent c) i( putMarker WHILE c( not c( frontIsClear c) c) w( turnRight w) i) ELSE e( move e) m)',
    'DEF run m( REPEAT R=2 r( IFELSE c( markersPresent c) i( pickMarker i) ELSE e( move turnLeft e) r) m)',
    'DEF run m( IFELSE c( frontIsClear c) i( move IF c( noMarkersPresent c) i( putMarker i) i) ELSE e( turnLeft e) m)',
    'DEF run m( WHILE c( leftIsClear c) w( turnRight WHILE c( markersPresent c) w( pickMarker w) w) m)',
    
    # cleanHouse1
    'DEF run m( WHILE c( noMarkersPresent c) w( IF c( leftIsClear c) i( turnLeft i) move IF c( markersPresent c) i( pickMarker i) w) m)',
    # fourCorners1
    'DEF run m( WHILE c( noMarkersPresent c) w( WHILE c( frontIsClear c) w( move w) IF c( noMarkersPresent c) i( putMarker turnLeft move i) w) m)',
    # harvester1
    'DEF run m( WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)',
    # maze
    'DEF run m( WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( turnRight i) ELSE e( WHILE c( not c( frontIsClear c) c) w( turnLeft w) e) move w) m)',
    # stairClimber1
    'DEF run m( WHILE c( noMarkersPresent c) w( turnLeft move turnRight move w) m)',
    # topOff1
    'DEF run m( WHILE c( frontIsClear c) w( IF c( markersPresent c) i( putMarker i) move w) m)'
]


#### We want to to see both of the policy being executed in the env: Best way is to let them do that.
all_tokens = ['DEF', 'run', 'm(', 'm)', 'move', 'turnRight',
              'turnLeft', 'pickMarker', 'putMarker', 'r(', 'r)', 'R=0', 'R=1', 'R=2',
              'R=3', 'R=4', 'R=5', 'R=6', 'R=7', 'R=8', 'R=9', 'R=10', 'R=11', 'R=12',
              'R=13', 'R=14', 'R=15', 'R=16', 'R=17', 'R=18', 'R=19', 'REPEAT', 'c(',
              'c)', 'i(', 'i)', 'e(', 'e)', 'IF', 'IFELSE', 'ELSE', 'frontIsClear',
              'leftIsClear', 'rightIsClear', 'markersPresent', 'noMarkersPresent',
              'not', 'w(', 'w)', 'WHILE']


def action_seq_to_tokens(act_seq: torch.Tensor) -> List[int]:

    act_seq = act_seq.squeeze(0).tolist()
    
    tokens = []
    token = None
    for action in act_seq:
        
        if action == 0: 
            # Move
            token = 4
        elif action == 1: 
            # Turn left
            token = 6
        elif action == 2: 
            # Turn Right
            token = 5
        elif action == 3:
            # Pick Marker 
            token = 7
        elif action == 4:
            # Pick Marker 
            token = 8
        tokens.append(token)
    return tokens

def extract_actions_from_tokens(tokens: List[int]) -> List[int]:

    raise Exception("Implementation is not right yet!")
    action_tokens = {4, 6, 5, 7, 8}
    actions = []
    for token in tokens:
        if token in action_tokens:
            actions.append(token)
    return actions

def get_neural_policy_act_sequence(model: nn.Module, env: EmptyTask,  z: torch.Tensor):

    neural_policy = partial(model.policy_executor, 
                                a_h_teacher_enforcing=False, 
                                a_h_mask=None,  
                                a_h = torch.zeros(size = (1, 10, 100)))
        
        # Start from:
    
    world = env.generate_state()
    state = world.s #torch.tensor(world.s dtype = torch.float32)#
    state = torch.tensor(np.moveaxis(state, [-2,-3,-1], [-1,-2,-3]), dtype = torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    act_seq = neural_policy(z = z, s_h = state)[0]

    return act_seq


def encode_program_token_to_latent(dsl, model, input_program_tokens):
    input_program = torch.tensor(dsl.pad_tokens(input_program_tokens, 45))
    z = model.encode_program(input_program)
    return z


def trace_programs(dsl, model, env, p: str):

    
    input_program_tokens = p
    z = encode_program_token_to_latent(dsl, model, input_program_tokens)

    # Policy executor: 
    act_seq = get_neural_policy_act_sequence(model, env, z)
    p_tokens = action_seq_to_tokens(act_seq)

    program_0 = dsl.parse_int_to_node(p_tokens)
    env.trace_program(program_0, image_name="gifs/Neuraltrace.gif", max_steps=200, label_text="Neural Policy", font_size=22)

    # Run the actual program:
    program = dsl.parse_int_to_node(p)
    env.trace_program(program, image_name="gifs/Programtrace.gif", max_steps=200, label_text="Program", font_size=22)



def main():

    ### LOAD MODEL
    dsl = DSL.init_default_karel()
    device = torch.device('cpu')
    model = LeapsVAEAttention(dsl, device)

    params = torch.load('output/LeapsVAETransformerMainRun/model/LeapsVAETransformerMainRun-best_val.ptp', map_location=device)
    model.load_state_dict(params, strict=False)

    env = EmptyTask(seed = 0)

    p = 'DEF run m( move turnRight move turnRight move turnRight move turnRight move turnRight m)'
    p = dsl.parse_str_to_int(p)



    trace_programs(dsl, model, env, p)

if __name__ == "__main__":
    main()