import argparse
from cfg_parser import cfg_parser
import os
import shutil
from pipeline import exp_pipeline
from utils import model_inp_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="MUM - Model Utilization Maximization", allow_abbrev=False
    )
    parser.add_argument(
        "-config",
        type=str,
        required=True,
        help="relative path of the file, relative to the cfg/ folder",
    )
    parser.add_argument(
        "-w", "--wandb", action="store_true", help="Whether to log to wandb or not"
    )
    parser.add_argument("-s", "--seed", default=0, help="Random seed")
    parser.add_argument(
        "-n", "--num_worker", type=int, default=10, help="No. of workers"
    )

    # Hyperparameters
    parser.add_argument("--lr", type=float, required=True, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, required=True, default=1e-4, help="weight decay")
    parser.add_argument("--arch", type=str, required=True, default="vgg11", help="name of architecture")
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("--bs", type=int, required=True, help="batch size")

    parser.add_argument("--ckpt", type=str, default=None, help="model path")
    parser.add_argument("--dataset", type= str, default='SVHN', help="Base dataset name")
    parser.add_argument("--source", type=str, default="svhn", help="name of source dataset")
    parser.add_argument("--target", type=str, default="mnist", help="name of target dataset")

    parser.add_argument("--method", type=str, required=True, choices=['st', 'sw', 'ta'],
                        help="Which method to run - st for source-train, sw for select-weights, ta for target-adapt")

    (args, unknown_args) = parser.parse_known_args()

    cfg_file = os.path.join("cfg", args.config)

    cfg = cfg_parser(cfg_file)
    model_cfg = cfg['model']
    exp_cfg = cfg['experiment']
    data_cfg = cfg['data']

    model_cfg['optimizer']['args']['lr'] = args.lr
    model_cfg['optimizer']['args']['weight_decay'] = args.wd
    model_cfg['architecture'] = args.arch
    model_cfg['epochs'] = args.epochs
    model_cfg['batch_size'] = args.bs
    model_cfg['method'] = args.method
    model_cfg['img_size'] = model_inp_size(model_cfg['architecture'])

    exp_cfg['load'] = args.ckpt
    exp_cfg['config'] = args.config
    exp_cfg['name'] = args.config.split('.json')[0]
    exp_cfg['wandb'] = args.wandb

    data_cfg['dataset'] = args.dataset
    data_cfg['source'] = args.source
    data_cfg['target'] = args.target

    if 'experiment_data' not in os.listdir():
        os.mkdir('experiment_data')

    sub_folder_name = data_cfg['dataset']+'_'+data_cfg['source']+'_'+data_cfg['target']+'_'+model_cfg['method']
    exp_cfg['log_folder'] = 'experiment_data/' + sub_folder_name
    if sub_folder_name in os.listdir('experiment_data/'):
        shutil.rmtree(exp_cfg['log_folder'])
        os.mkdir(exp_cfg['log_folder'])
    else:
        os.mkdir(exp_cfg['log_folder'])

    cfg = {'model': model_cfg, 'data': data_cfg, 'exp': exp_cfg}

    exp_pipeline(cfg)

