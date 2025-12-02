import torch
# from dsl.parser import Parser
from dsl import DSL
from vae.models.leaps_vae import LeapsVAE
from config import Config


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


if __name__ == '__main__':

    dsl = DSL.init_default_karel()
    device = torch.device('cpu')
    model = LeapsVAE(dsl, device)

    params = torch.load(f'params/leaps_vae_{256}.ptp', map_location=device)
    model.load_state_dict(params, strict=False)
    
    results = []
    for i, p in enumerate(PROGRAMS):

        input_program_tokens = dsl.parse_str_to_int(p)
        input_program = torch.tensor(dsl.pad_tokens(input_program_tokens, 45))
        # print(p)
        
        z = model.encode_program(input_program)
        pred_progs = model.decode_vector(z)
        #print(pred_progs)
        
        output_program = dsl.parse_int_to_str(pred_progs[0])
        # print(output_program)

        # print('embedding space:', z.detach().cpu().numpy().tolist(), 'shape:', z.shape)
        results.append(output_program == p)
        # print(output_program)
    
    print(results)
        