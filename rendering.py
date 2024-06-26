import torch


def get_ndc_range(pixels):
    """Returns the Normalized Device Coordinate (NDC) range  (-1, 1) for given pixels.

    Learn more:
    https://www.khronos.org/opengl/wiki/Coordinate_Transformations
    """
    pixels = pixels
    half_pixels = pixels / 2.0

    pixel_space = torch.arange(start=0, end=pixels, step=1.0) + 0.5
    # Bring (0, range) -> (-1, 1)
    ndc_space = (pixel_space - half_pixels) / half_pixels

    return ndc_space


def ndc_pixel_coordinates(width, height):
    """Returns the NDC pixel coordinate, a tensor [H,W,2]

    As input we specific the width and height in pixel dimensions.
    """
    # Make sure width and height are integers even if in float format.
    assert isinstance(width, int) and isinstance(height, int)

    # NDC space is normalized between (-1, 1) in all 3 dimensions,
    # Pixel coordinates are defined from the center of the pixel hence the + 0.5
    x = get_ndc_range(width)
    y = get_ndc_range(height)
    yy, xx = torch.meshgrid(x, y, indexing="ij")

    # NDC pixel coordinates (H, W, 2)
    ndc_grid = torch.stack([xx, yy], axis=-1)

    return ndc_grid


def render_alpha_blend(width, height, means, covariances, opacities):
    """Renders an image from a set of Gaussians with alpha blending.

    Gaussians are assumed to be already in camera-space.
    """

    ndc_grid = ndc_pixel_coordinates(width, height).to(means.device)  # [H, W, 2]

    n_gaussians = means.shape[0]
    xy_means = means[:, :2]

    ndc_grid = ndc_grid.repeat(n_gaussians, 1, 1, 1)
    d = ndc_grid - xy_means[:, None, None, :]

    v = d[..., None, :] @ torch.linalg.inv(covariances)[:, None, None, :, :] @ d[..., None]
    v = v.squeeze()
    v = torch.exp(-(1 / 2.0) * v)

    alpha = v * opacities[:, None, :]
    image = torch.sum(alpha, dim=0)
    return image
