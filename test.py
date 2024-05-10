import argparse
import os
from glob import glob

from PIL import Image
import numpy as np
import torch

import utils
from models.transformation import Transformer
from models.biggan import Generator
from models.emonet import emonet


def main(args):

    # Choose GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder

    logger = utils.create_logger(args.results_dir)
    logger.info(f"Experiment directory created at {args.results_dir}")

    # Some characteristics
    # --------------------------------------------------------------------------------------------------------------
    dim_z = {
        128: 120,
        256: 140,
        512: 128
    }.get(args.generator_pixels)

    vocab_size = 1000
    categories_file = "categories_imagenet.txt"
    categories = [x.strip() for x in open(categories_file)]

    # Setting up Transformer
    # --------------------------------------------------------------------------------------------------------------
    model = Transformer(dim_z, args.transformer == "OneDirection")
    model = model.to(device)

    assert os.path.isfile(args.ckpt), f'Could not find DiT checkpoint at {args.ckpt}'
    
    model_state, _, _ = utils.update_model_state(model.state_dict(), args.ckpt, testing=True)
    model.load_state_dict(model_state)
    model.eval()

    print("Model Checkpoint loaded successfully")

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

    def make_image(z, y, step_size, transform):
        if transform:
            z_transformed = model(z, y, step_size)
            z_transformed = z.norm() * z_transformed / z_transformed.norm()
            z = z_transformed

        gan_images = utils.denorm(generator(z, y))
        gan_images_np = gan_images.permute(0, 2, 3, 1).detach().cpu().numpy()
        gan_images = input_transform(gan_images)
        gan_images = gan_images.view(-1, *gan_images.shape[-3:])
        gan_images = gan_images.to(device)

        out_scores_current = output_transform(assessor(gan_images))
        out_scores_current = out_scores_current.detach().cpu().numpy()
        if len(out_scores_current.shape) == 1:
            out_scores_current = np.expand_dims(out_scores_current, 1)

        return(gan_images_np, z, out_scores_current)


    # Test settings
    num_samples = 10
    truncation = args.test_truncation
    iters = 3
    np.random.seed(seed=999)
    annotate = True

    num_categories = 1 if vocab_size == 0 else vocab_size

    for y in range(num_categories):

        ims = []
        outscores = []

        zs = utils.truncated_z_sample(num_samples, dim_z, truncation)
        ys = np.repeat(y, num_samples)
        zs = torch.from_numpy(zs).type(torch.FloatTensor).to(device)
        ys = torch.from_numpy(ys).to(device)
        ys = utils.one_hot(ys, vocab_size)
        step_sizes = np.repeat(np.array(args.alpha), num_samples * dim_z).reshape([num_samples, dim_z])
        step_sizes = torch.from_numpy(step_sizes).type(torch.FloatTensor).to(device)
        feed_dicts = []

        for batch_start in range(0, num_samples, 4):
            s = slice(batch_start, min(num_samples, batch_start + 4))
            feed_dicts.append({"z": zs[s], "y": ys[s], "truncation": truncation, "step_sizes": step_sizes[s]})

        for feed_dict in feed_dicts:
            ims_batch = []
            outscores_batch = []
            z_start = feed_dict["z"]
            step_sizes = feed_dict["step_sizes"]

            if args.mode == "iterative":
                logger.info("iterative")

                # original seed image
                x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=False)
                x = np.uint8(x)
                if annotate:
                    ims_batch.append(utils.annotate_outscore(x, outscore))
                else:
                    if annotate:
                        ims_batch.append(utils.annotate_outscore(x, outscore))
                    else:
                        ims_batch.append(x)
                outscores_batch.append(outscore)

                # negative clone images
                z_next = z_start
                step_sizes = -step_sizes
                for iter in range(0, iters, 1):
                    feed_dict["step_sizes"] = step_sizes
                    feed_dict["z"] = z_next
                    x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True)
                    x = np.uint8(x)
                    z_next = tmp
                    if annotate:
                        ims_batch.append(utils.annotate_outscore(x, outscore))
                    else:
                        if annotate:
                            ims_batch.append(utils.annotate_outscore(x, outscore))
                        else:
                            ims_batch.append(x)
                    outscores_batch.append(outscore)

                ims_batch.reverse()

                # positive clone images
                step_sizes = -step_sizes
                z_next = z_start
                for iter in range(0, iters, 1):
                    feed_dict["step_sizes"] = step_sizes
                    feed_dict["z"] = z_next

                    x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True)
                    x = np.uint8(x)
                    z_next = tmp

                    if annotate:
                        ims_batch.append(utils.annotate_outscore(x, outscore))
                    else:
                        ims_batch.append(x)
                    outscores_batch.append(outscore)

            else:
                logger.info("bigger_step")

                # original seed image
                x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=False)
                x = np.uint8(x)
                if annotate:
                    ims_batch.append(utils.annotate_outscore(x, outscore))
                else:
                    ims_batch.append(x)
                outscores_batch.append(outscore)

                # negative clone images
                step_sizes = -step_sizes
                for iter in range(0, iters, 1):
                    feed_dict["step_sizes"] = step_sizes * (iter + 1)

                    x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True)
                    x = np.uint8(x)

                    if annotate:
                        ims_batch.append(utils.annotate_outscore(x, outscore))
                    else:
                        ims_batch.append(x)
                    outscores_batch.append(outscore)

                ims_batch.reverse()
                outscores_batch.reverse()

                # positive clone images
                step_sizes = -step_sizes
                for iter in range(0, iters, 1):
                    feed_dict["step_sizes"] = step_sizes * (iter + 1)

                    x, tmp, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True)
                    x = np.uint8(x)
                    if annotate:
                        ims_batch.append(utils.annotate_outscore(x, outscore))
                    else:
                        ims_batch.append(x)
                    outscores_batch.append(outscore)

            ims_batch = [np.expand_dims(im, 0) for im in ims_batch]
            ims_batch = np.concatenate(ims_batch, axis=0)
            ims_batch = np.transpose(ims_batch, (1, 0, 2, 3, 4))
            ims.append(ims_batch)

            outscores_batch = [np.expand_dims(outscore, 0) for outscore in outscores_batch]
            outscores_batch = np.concatenate(outscores_batch, axis=0)
            outscores_batch = np.transpose(outscores_batch, (1, 0, 2))
            outscores.append(outscores_batch)

        ims = np.concatenate(ims, axis=0)
        outscores = np.concatenate(outscores, axis=0)
        ims_final = np.reshape(ims, (ims.shape[0] * ims.shape[1], ims.shape[2], ims.shape[3], ims.shape[4]))
        I = Image.fromarray(utils.imgrid(ims_final, cols=iters * 2 + 1))
        I.save(os.path.join(args.results_dir, categories[y] + ".jpg"))
        print("y: ", y)


if __name__ == "__main__":
    # Collect command line arguments
    # --------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, required=True)
    parser.add_argument('--results-dir', type=str, default="results", help='directory to store model results')
    parser.add_argument('--alpha', type=float, default=0.1, help='stepsize for testing')
    parser.add_argument('--test_truncation', type=float, default=1, help='truncation to use in test phase')
    parser.add_argument('--generator-pixels', type=int, default=256, help='generator function to use')
    parser.add_argument('--assessor', type=str, default="emonet", help='assessor function to compute the image property of interest')
    parser.add_argument('--transformer', default="OneDirection", type=str, help="transformer function")
    parser.add_argument('--mode', default="bigger_step", choices=["iterative", "bigger_step"],
                    help="how to make the test sequences. bigger_step was used in the paper.")

    args = parser.parse_args()

    main(args)