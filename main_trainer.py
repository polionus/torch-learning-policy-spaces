import torch
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from utils.aim_utils import get_run_config_dict
from vae.program_dataset import make_dataloaders
from vae.trainer import Trainer
from aim import Run

if __name__ == '__main__':

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsl = DSL.init_default_karel()

    model = load_model(Config.model_name, dsl, device)

    p_train_dataloader, p_val_dataloader, _ = make_dataloaders(dsl, device)

    run = Run(experiment=f"{Config.experiment_name}")
    run['hparams'] = get_run_config_dict()
    trainer = Trainer(model, run)

    StdoutLogger.log('Main', f'Starting trainer for model {Config.model_name}')

    trainer.train(p_train_dataloader, p_val_dataloader)
    
    StdoutLogger.log('Main', 'Trainer finished.')
