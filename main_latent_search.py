import torch
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from search.latent_search import LatentSearch
from tasks import get_task_cls
from aim import Run


if __name__ == '__main__':
    
    dsl = DSL.init_default_karel()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(Config.model_name, dsl, device)
    
    task_cls = get_task_cls(Config.env_task)
    
    params = torch.load(Config.model_params_path, map_location=device)
    model.load_state_dict(params, strict=False)

    run = Run(experiment=Config.experiment_name)
    config = {
        'population_size': Config.search_population_size,
        'reduce_to_mean': Config.search_reduce_to_mean,
        'sigma': Config.search_sigma,
        'number_of_iterations': Config.search_number_iterations,
        'number_of_executions': Config.search_number_executions,
    }
    run['hparams'] = config
    
    searcher = LatentSearch(model, task_cls, dsl, run)
    


    StdoutLogger.log('Main', f'Starting Latent Search with model {Config.model_name} for task {Config.env_task}')
    
    best_program, converged, num_evaluations = searcher.search()
    
    StdoutLogger.log('Main', f'Converged: {converged}')
    StdoutLogger.log('Main', f'Final program: {best_program}')
    StdoutLogger.log('Main', f'Number of evaluations: {num_evaluations}')
