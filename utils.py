import torch
import numpy as np
import logging
import os

from scipy.stats import truncnorm
import PIL.ImageDraw
import PIL.ImageFont

def one_hot(index, vocab_size=1000):
    output = torch.zeros(index.size(0), vocab_size).to(index.device)
    output.scatter_(1, index.unsqueeze(-1), 1)
    return output

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1) * 255

def truncated_z_sample(batch_size, dim_z, truncation=1):
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z))
    return truncation * values

def imgrid(imarray, cols=5, pad=1):
    if imarray.dtype != np.uint8:
        imarray = np.uint8(imarray)
        # raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows * H, cols * W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid

def annotate_outscore(array, outscore):
    for i in range(array.shape[0]):
        I = PIL.Image.fromarray(np.uint8(array[i,:,:,:]))
        draw = PIL.ImageDraw.Draw(I)
        font =  PIL.ImageFont.truetype("arial.ttf", int(array.shape[1]/8.5))
        message = str(round(np.squeeze(outscore)[i], 2))
        x, y = (0, 0)
        w, h = font.getsize(message)
        #print(w, h)
        draw.rectangle((x, y, x + w, y + h), fill='white')
        draw.text((x, y), message, fill="black", font=font)
        array[i, :, :, :] = np.array(I)
    return(array)

def update_model_state(model_state, state_dict_path, fine_tuning=False, testing=False):

    pretrained_state = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
    ckpt_steps = 0

    if fine_tuning:
        opt_state = None
    else:
        opt_state = pretrained_state["opt"]
        pretrained_state = pretrained_state["model"]
        
        if not testing:
            path = state_dict_path.split(os.sep)[-1].split("-")
            ckpt_steps = int(path[-1][0:-3])

    for name, param in pretrained_state.items():
        if name not in model_state:
                continue
        elif isinstance(param, torch.nn.parameter.Parameter):
            param = param.data

        if fine_tuning:
            '''
                Load all weights but AdaLN; 
                Freeze all weights but Bias and LayerNorm
            '''
            if "adaLN" not in name:
                model_state[name].copy_(param)
            if "bias" not in name or "adaLN" not in name:
                model_state[name].requires_grad = False
        else:
            model_state[name].copy_(param)
    
    return model_state, opt_state, ckpt_steps


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    
    return logger