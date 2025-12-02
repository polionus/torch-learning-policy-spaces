from config import Config
from typing import Dict

def get_run_config_dict() -> Dict:

    
    config = {
        "learning_rate": Config.trainer_optim_lr,
        "a_h_loss_coeff": Config.trainer_a_h_loss_coef,
        "prog_loss_coeff": Config.trainer_prog_loss_coef,
        "latent_loss_coeff": Config.trainer_latent_loss_coef,
        "num_epochs": Config.trainer_num_epochs,
        "a_h_teacher_enforcing": not Config.trainer_disable_a_h_teacher_enforcing,
        "prog_teacher_enforcing": not Config.trainer_disable_prog_teacher_enforcing,
        "ratio_train": Config.data_ratio_train,
        "ratio_test": Config.data_ratio_test,
        "ratio_val": Config.data_ratio_val,
        "batch_size": Config.data_batch_size,
    }
    return config
