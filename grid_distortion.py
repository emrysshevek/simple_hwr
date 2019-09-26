import cv2
import numpy as np
from scipy.interpolate import griddata
import sys

INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC
}
cv2.setNumThreads(0)

def occlude(img, occlusion_size=1, occlusion_freq=.5, occlusion_level=1, logger=None, noise_type=None):
    if occlusion_freq:
        return _occlude(img, occlusion_size, occlusion_freq, occlusion_level, logger)
    else:
        if noise_type is None:
            noise_type = "gaussian"
        return noise(img, occlusion_level=occlusion_level, logger=logger, noise_type=noise_type)

def _occlude(img, occlusion_size=1, occlusion_freq=.5, occlusion_level=1, logger=None):
    """
        Occlusion frequency : between 0% and this number will be occluded
        Occlusion level: maximum occlusion change (multiplier); each pixel to be occluded has a random occlusion probability;
                         then it is multiplied/divided by at most the occlusion level

        NOT IMPLEMENTED:
        OTHER OPTIONS:
            RANDOM OCCLUSION THRESHOLD
            RANDOM OCCLUSION LEVEL (within range)
            OCCLUSION SIZE
    Args:
        img:
        occlusion_size:
        occlusion_freq:
        occlusion_level: just "dim" these pixels a random amount; 1 - white, 0 - original image
        occlusion
        logger:

    Returns:

    """
    # Randomly choose occlusion frequency between 0 and specified occlusion
    # H X W X Channel
    random_state = np.random.RandomState()
    occlusion_freq = random_state.uniform(0, occlusion_freq) #
    binary_mask = random_state.choice(2, img.shape, p=[occlusion_freq, 1-occlusion_freq])
    #logger.debug(binary_mask)
    if occlusion_level==1:
        occlusion = np.where(binary_mask==0, 255, img) # replace 0's with white
    else: # 1 = .3
        sd = occlusion_level / 2 # ~95% of observations will be less extreme; if occlusion_level=1, we set so 95% of multipliers are <1
        random_mask = random_state.randn(*img.shape, ) * sd # * 2 - occlusion_level # min -occlusion, max occlusion
        random_mask = np.clip(random_mask, -1, 1)
        if False: # randomly whiten to different levels
            occlusion = np.where(binary_mask == 0, (1-random_mask)*img+255*random_mask, img)  # replace 0's with white
        else: # random noise
            random_mask = np.minimum((random_mask + 1) * img, 255)
            occlusion = np.where(binary_mask == 0, random_mask, img)
    return occlusion

# def noise(img, occlusion_size=1, occlusion_freq=.5, occlusion_level=1, logger=None):
#     """
#         NOT IMPLEMENTED:
#         OTHER OPTIONS:
#             RANDOM OCCLUSION THRESHOLD
#             RANDOM OCCLUSION LEVEL (within range)
#             OCCLUSION SIZE
#     Args:
#         img:
#         occlusion_size:
#         occlusion_freq:
#         occlusion_level: just "dim" these pixels a random amount; 1 - white, 0 - original image
#         occlusion
#         logger:
#
#     Returns:
#
#     """
#
#
#     # Randomly choose occlusion frequency between 0 and specified occlusion
#     # H X W X Channel
#     random_state = np.random.RandomState()
#     occlusion_freq = random_state.uniform(0, occlusion_freq)
#     binary_mask = random_state.choice(2, img.shape, p=[occlusion_freq, 1-occlusion_freq])
#     #logger.debug(binary_mask)
#     if occlusion_level==1:
#         occlusion = np.where(binary_mask==0, 255, img) # replace 0's with white
#     else:
#         random_mask = random_state.rand(*img.shape) * occlusion_level # occlude between not-at-all and occlusion-level
#         occlusion = np.where(binary_mask == 0, (1-random_mask)*img+255*random_mask, img)  # replace 0's with white
#
#     return occlusion

def warp_image(img, random_state=None, **kwargs):
    if random_state is None:
        random_state = np.random.RandomState()

    w_mesh_interval = kwargs.get('w_mesh_interval', 25)
    w_mesh_std = kwargs.get('w_mesh_std', 3.0)

    h_mesh_interval = kwargs.get('h_mesh_interval', 25)
    h_mesh_std = kwargs.get('h_mesh_std', 3.0)

    interpolation_method = kwargs.get('interpolation', 'linear')

    h, w = img.shape[:2]

    if kwargs.get("fit_interval_to_image", True):
        # Change interval so it fits the image size
        w_ratio = w / float(w_mesh_interval)
        h_ratio = h / float(h_mesh_interval)

        w_ratio = max(1, round(w_ratio))
        h_ratio = max(1, round(h_ratio))

        w_mesh_interval = w / w_ratio
        h_mesh_interval = h / h_ratio
        ############################################

    # Get control points
    source = np.mgrid[0:h+h_mesh_interval:h_mesh_interval, 0:w+w_mesh_interval:w_mesh_interval]
    source = source.transpose(1,2,0).reshape(-1,2)

    if kwargs.get("draw_grid_lines", False):
        if len(img.shape) == 2 or img.shape[2]==1: # if already grayscale
            color = 0
        else:
            color = np.array([0,0,255])
        for s in source:
            img[int(s[0]):int(s[0])+1,:] = color
            img[:,int(s[1]):int(s[1])+1] = color

    # Perturb source control points
    destination = source.copy()
    source_shape = source.shape[:1]
    destination[:,0] = destination[:,0] + random_state.normal(0.0, h_mesh_std, size=source_shape)
    destination[:,1] = destination[:,1] + random_state.normal(0.0, w_mesh_std, size=source_shape)

    # Warp image
    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(destination, source, (grid_x, grid_y), method=interpolation_method).astype(np.float32)
    map_x = grid_z[:,:,1]
    map_y = grid_z[:,:,0]
    warped = cv2.remap(img, map_x, map_y, INTERPOLATION[interpolation_method], borderValue=(255,255,255))

    return warped

def noise(img, occlusion_level=1, logger=None, noise_type="gaussian"):
    if noise_type == "gaussian":
        return gaussian_noise(img, occlusion_level=occlusion_level, logger=logger)
    else:
        raise Exception("Not implemented")

def gaussian_noise(img, occlusion_level=1, logger=None):
    """
        occlusion_level: .1 - light haze, 1 heavy

    """

    random_state = np.random.RandomState()
    sd = occlusion_level / 2  # ~95% of observations will be less extreme; if occlusion_level=1, we set so 95% of multipliers are <1
    noise_mask = random_state.randn(*img.shape, ) * sd  # * 2 - occlusion_level # min -occlusion, max occlusion
    noise_mask = np.clip(noise_mask, -1, 1) * 255/2
    noisy_img = np.clip(img + noise_mask, 0, 255)
    return noisy_img

    # elif noise_typ == "s&p":
    #     row, col, ch = image.shape
    #     s_vs_p = 0.5
    #     amount = 0.004
    #     out = image
    #     # Salt mode
    #     num_salt = np.ceil(amount * image.size * s_vs_p)
    #     coords = [np.random.randint(0, i - 1, int(num_salt))
    #               for i in image.shape]
    #     out[coords] = 1
    #
    #     # Pepper mode
    #     num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    #     coords = [np.random.randint(0, i - 1, int(num_pepper))
    #               for i in image.shape]
    #     out[coords] = 0
    #     return out
    # elif noise_typ == "poisson":
    #     vals = len(np.unique(image))
    #     vals = 2 ** np.ceil(np.log2(vals))
    #     noisy = np.random.poisson(image * vals) / float(vals)
    #     return noisy
    # elif noise_typ == "speckle":
    #     row, col, ch = image.shape
    #     gauss = np.random.randn(row, col, ch)
    #     gauss = gauss.reshape(row, col, ch)
    #     noisy = image + image * gauss
    #     return noisy


def test():
    import matplotlib.pylab as plt
    if False:
        input_image = sys.argv[1]
        output_image = sys.argv[2]
        img = cv2.imread(input_image)
        cv2.imwrite(output_image, img)
    else:
        input_imge = "/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/lines/m04/m04-061/m04-061-02.png"
        img = cv2.imread(input_imge,0)
        img = noise(img, occlusion_level=.5)
        plt.imshow(img, cmap="gray")
        plt.show()
if __name__ == "__main__":
    test()


