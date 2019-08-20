import cv2
import numpy as np
from scipy.interpolate import griddata
import sys

INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC
}
cv2.setNumThreads(0)

def occlude(img, occlusion_size=1, occlusion_freq=.5, logger=None):
    # Randomly choose occlusion frequency between 0 and specified occlusion
    # H X W X Channel
    random_state = np.random.RandomState()
    occlusion_freq = random_state.uniform(0, occlusion_freq)
    binary_mask = random_state.choice(2, img.shape, p=[occlusion_freq, 1-occlusion_freq])
    #logger.debug(binary_mask)
    occlusion = np.where(binary_mask==0, 255, img) # replace 0's with white
    return occlusion

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

if __name__ == "__main__":
    input_image = sys.argv[1]
    output_image = sys.argv[2]
    img = cv2.imread(input_image)
    #img = cv2.imread("/media/data/GitHub/simple_hwr/data/prepare_IAM_Lines/lines/m04/m04-061/m04-061-02.png",0)
    img = warp_image(img, draw_grid_lines=True)
    cv2.imwrite(output_image, img)

