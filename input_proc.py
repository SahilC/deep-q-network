import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image

resize = T.Compose([T.ToPILImage(),T.Scale(40, interpolation=Image.CUBIC), T.ToTensor()])
screen_width = 600

def get_cart_location(env):
	world_width = env.x_threshold*2
	scale = screen_width/world_width
	return(env.state[0]*scale + screen_width/2.0)

def get_screen(env):
    screen = env.render(mode='rgb_array').transpose((2,0,1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location(env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width / 2):
        slice_range = slice(-view_width,None)
    else:
        slice_range = slice(int(cart_location - view_width / 2), int(cart_location + view_width / 2))
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)
