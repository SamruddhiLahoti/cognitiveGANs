import argparse
import os
from glob import glob

import numpy as np
import torch

import utils
from models.transformation import Transformer
from models.biggan import Generator
from models.emonet import emonet


def main(args):

    # Choose GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.log_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logger = utils.create_logger(args.log_dir)

    # Some characteristics
    # --------------------------------------------------------------------------------------------------------------
    dim_z = {
        128: 120,
        256: 140,
        512: 128
    }.get(args.generator_pixels)

    vocab_size = 1000

    # Setting up Transformer
    # --------------------------------------------------------------------------------------------------------------

    model = Transformer(dim_z, args.transformer == "OneDirection")
    model = model.to(device)

    # Setting up Generator
    # --------------------------------------------------------------------------------------------------------------
    model_path = f"saved_models/biggan-{args.generator_pixels}.pth"
    generator = Generator(code_dim=dim_z, pixels=args.generator_pixels)
    generator.load_state_dict(torch.load(model_path))

    for p in generator.parameters():
        p.requires_grad = False
    generator.eval()
    generator = generator.to(device)

    # Setting up Assessor
    # --------------------------------------------------------------------------------------------------------------
    parameters = torch.load("saved_models/EmoNet_valence_moments_resnet50_5_best.pth.tar", map_location='cpu')
    assessor_elements = emonet(True, parameters)
    if isinstance(assessor_elements, tuple):
        assessor = assessor_elements[0]
        input_transform = assessor_elements[1]
        output_transform = assessor_elements[2]
    else:
        assessor = assessor_elements

        def input_transform(x):
            return x  # identity, no preprocessing

        def output_transform(x):
            return x  # identity, no postprocessing

    if hasattr(assessor, 'parameters'):
        for p in assessor.parameters():
            p.requires_grad = False
    assessor.eval()
    assessor.to(device)

    # Training
    # --------------------------------------------------------------------------------------------------------------
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    criterion = torch.nn.MSELoss()

    ckpt_steps = 0 # steps loaded from checkpoint
    if args.ckpt is not None:
        assert os.path.isfile(args.ckpt), f'Could not find DiT checkpoint at {args.ckpt}'
        
        model_state, opt_state, ckpt_steps = utils.update_model_state(model.state_dict(), args.ckpt, fine_tuning=args.fine_tuning)
        model.load_state_dict(model_state)

        print("Model Checkpoint loaded successfully")

    #  training settings
    train_steps = 0
    batch_size = 4

    # create training set
    np.random.seed(seed=0)
    truncation = 1
    zs = utils.truncated_z_sample(args.num_samples, dim_z, truncation)
    ys = np.random.randint(0, vocab_size, size=zs.shape[0])

    logger.info("Starting training")
    save_path = f"{args.checkpoint_dir}/biggan{args.generator_pixels}-{args.assessor}"

    # loop over data batches
    for batch_start in range(0, args.num_samples, batch_size):

        # zero the parameter gradients
        optimizer.zero_grad()

        # skip batches we've already done (this would happen when resuming from a checkpoint)
        if batch_start <= ckpt_steps and ckpt_steps != 0:
            train_steps = train_steps + 1
            continue

        # input batch
        s = slice(batch_start, min(args.num_samples, batch_start + batch_size))
        z = torch.from_numpy(zs[s]).type(torch.FloatTensor).to(device)
        y = torch.from_numpy(ys[s]).to(device)
        step_sizes = (args.train_alpha_b - args.train_alpha_a) * \
            np.random.random(size=(batch_size)) + args.train_alpha_a  # sample step_sizes
        step_sizes_broadcast = np.repeat(step_sizes, dim_z).reshape([batch_size, dim_z])
        step_sizes_broadcast = torch.from_numpy(step_sizes_broadcast).type(torch.FloatTensor).to(device)

        # ganalyze steps
        gan_images = generator(z, utils.one_hot(y))
        gan_images = input_transform(utils.denorm(gan_images))
        gan_images = gan_images.view(-1, *gan_images.shape[-3:])
        gan_images = gan_images.to(device)
        out_scores = output_transform(assessor(gan_images)).to(device).float()
        target_scores = out_scores + torch.from_numpy(step_sizes).to(device).float()

        z_transformed = model(z, utils.one_hot(y), step_sizes_broadcast)
        gan_images_transformed = generator(z_transformed, utils.one_hot(y))
        gan_images_transformed = input_transform(utils.denorm(gan_images_transformed))
        gan_images_transformed = gan_images_transformed.view(-1, *gan_images_transformed.shape[-3:])
        gan_images_transformed = gan_images_transformed.to(device)
        out_scores_transformed = output_transform(assessor(gan_images_transformed)).to(device).float()

        # compute loss
        loss = criterion(out_scores_transformed, target_scores)

        # backwards
        loss.backward()
        optimizer.step()

        if train_steps % 250 == 0:
            checkpoint = {
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "args": args
            }
            checkpoint_path = f"{save_path}-{(train_steps + ckpt_steps):07d}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        train_steps = train_steps + 1

    checkpoint = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "args": args
    }
    checkpoint_path = f"{save_path}-{(train_steps + ckpt_steps):07d}.pth"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    # Collect command line arguments
    # --------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default="log", help='directory for the log file')
    parser.add_argument('--checkpoint-dir', type=str, default="saved_models", help='directory to save model ckpt')
    parser.add_argument('--num-samples', type=int, default=400000, help='number of samples to train for')
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument('--train-alpha-a', type=float, default=-0.5, help='lower limit for step sizes to use during training')
    parser.add_argument('--train-alpha-b', type=float, default=0.5, help='upper limit for step sizes to use during training')
    parser.add_argument('--generator-pixels', type=int, default=256, help='generator function to use')
    parser.add_argument('--assessor', type=str, default="emonet", help='assessor function to compute the image property of interest')
    parser.add_argument('--transformer', default="OneDirection", type=str, help="transformer function")

    args = parser.parse_args()

    main(args)