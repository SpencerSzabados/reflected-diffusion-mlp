"""
    File contains helper functions for laoding and saving model checkpoints.
"""


import os 
import torch as th
from . import logger
from . import distribute_util
import torch.distributed as dist


def obtain_slurm_ckpt_dir():
    SLURM_JOB_ID = os.environ['SLURM_JOB_ID']
    USER = os.environ['USER']
    ckpt_dir=f'/checkpoint/{USER}/{SLURM_JOB_ID}'
    return ckpt_dir


def _parse_resume_step_from_filename(filename):
        """
        Parse filenames of the form path/to/checkpoint_NNNNNN.pt, where NNNNNN is the
        checkpoint's number of steps.
        """
        split = filename.split("checkpoint_")
        if len(split) == 2:
            split1 = split[-1].split(".")[0]
            try:
                return int(split1)
            except ValueError:
                return 0
        else:
            return 0


def find_resume_checkpoint_aux(ckpt_dir):
    # search the lastest checkpoint in ckpt_dir
    if not os.path.exists(ckpt_dir):
        logger.warn(f'CANNOT FIND: {ckpt_dir}')
        return None

    last_resume_step = -1
    for filename in os.listdir(ckpt_dir):
        if ('checkpoint' in filename) and ('.pth' in filename):
            resume_step = _parse_resume_step_from_filename(filename)
            if resume_step > last_resume_step:
                last_resume_step = resume_step

    if last_resume_step > -1:
        return os.path.join(ckpt_dir, f'checkpoint_{last_resume_step}.pth')
    else:
        return None


def find_resume_checkpoint(args):
    try:
        ckpt_dir = os.path.join(args.working_dir, f"exps/{args.exps}/")
        ckpt_file_path = find_resume_checkpoint_aux(ckpt_dir)
        logger.info(f'ckpt_file_path is set to {ckpt_file_path}')
        return ckpt_file_path

    except KeyError:
        logger.info('Cannot detect checkpoint dir.')
        return None


def load_and_sync_parameters(args, model, ema_model=None, optimizer=None):
    """
    Loads model, ema, and optimizer states from checkpoints.
    """
    resume_step = 0
    if args.resume_training:
        if args.resume_checkpoint is not "":
            resume_checkpoint = args.resume_checkpoint
        else:
            resume_checkpoint =  find_resume_checkpoint(args)

        if resume_checkpoint is not None:
            logger.log(f"Found checkpoint: {resume_checkpoint}")
            resume_step = _parse_resume_step_from_filename(resume_checkpoint)
            resume_checkpoint = th.load(resume_checkpoint, map_location=distribute_util.dev(), weights_only=False)
            if dist.get_rank() == 0:
                logger.log(f"Loading model state...")
                model.load_state_dict(resume_checkpoint["model"])
                if ema_model is not None:
                    logger.log(f"Loading ema_model state...")
                    ema_model.load_state_dict(resume_checkpoint["ema_model"])
                if optimizer is not None:
                    logger.log(f"Loading optimizer state...")
                    optimizer.load_state_dict(resume_checkpoint["optimizer"])
            
    model = model.to(device=distribute_util.dev())
    
    if ema_model is not None:
        ema_model = ema_model.to(device=distribute_util.dev())

        return resume_step, model, ema_model, optimizer
    else:
        return resume_step, model


def save_checkpoint(outdir, model, ema_model, optimizer, step):
    logger.log("Saving model...")
    checkpoint = {"step": step,
                  "model": model.state_dict(),
                  "ema_model": ema_model.state_dict(),
                  "optimizer": optimizer.state_dict()}
    th.save(checkpoint, f"{outdir}/checkpoint_{step}.pth")