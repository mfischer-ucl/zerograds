import torch
import torch.nn.functional as F

# splatting-based caustics, inspired by https://madebyevan.com/webgl-water/


def grid(N, device):
    return torch.stack(torch.meshgrid(torch.linspace(-1, 1, N),
                                      torch.linspace(-1, 1, N)), -1).to(device)


def intersect_plane(origins, directions, plane_normal, plane_d=0):
    dot_nd = torch.sum(directions * plane_normal, dim=2, keepdim=True)
    valid = dot_nd != 0
    p0l0 = plane_d - torch.sum(origins * plane_normal, dim=2, keepdim=True)
    t = p0l0 / dot_nd
    t = torch.where(valid, t, torch.zeros_like(t))
    intersection_points = origins + t * directions
    return intersection_points


def refract(ray_directions, surface_normals, ior=1.33):
    # no need to normalize: normals have been normalized before, ray directions are normalized by design
    # ray_dirs = torch.nn.functional.normalize(ray_directions, dim=2)
    # normals = torch.nn.functional.normalize(surface_normals, dim=2)

    # Calculate cosines of angles
    cos_theta_i = -torch.sum(ray_directions * surface_normals, dim=2, keepdim=True)
    cos_theta_t2 = 1 - ior ** 2 * (1 - cos_theta_i ** 2)

    # Handle total internal reflection
    total_internal_reflection = cos_theta_t2 < 0
    cos_theta_t2 = torch.where(total_internal_reflection, torch.zeros_like(cos_theta_t2), cos_theta_t2)

    # Refracted direction calculation
    r_out_perp = ior * (ray_directions + cos_theta_i * surface_normals)
    r_out_parallel = -torch.sqrt(torch.abs(cos_theta_t2)) * surface_normals
    refracted = r_out_perp + r_out_parallel

    # Handle total internal reflection
    refracted = torch.where(total_internal_reflection.repeat(1, 1, 3), torch.zeros_like(refracted), refracted)
    refracted_norm = F.normalize(refracted, p=2, dim=-1)
    return refracted_norm


def splat(points, c_res):
    image = torch.zeros(c_res, c_res, device=points.device)  # cannot use expand because tensor not read-only
    mask = (points[:, :, 0] > -1) & (points[:, :, 0] < 1) & \
           (points[:, :, 1] > -1) & (points[:, :, 1] < 1)
    valid_points = ((points[mask] * 0.5 + 0.5) * c_res - 1).long()
    return add_at_indices_torch(image, valid_points)


def add_at_indices_torch(image, pixel_indices):
    image_flat = image.flatten()

    # Convert the 2D pixel_indices into 1D indices
    # Assuming pixel_indices[:, 0] are row indices and pixel_indices[:, 1] are column indices
    indices_flat = pixel_indices[:, 0] * image.shape[1] + pixel_indices[:, 1]

    # Using scatter_add to accumulate at indices adding 1 at each index
    ones = torch.ones_like(indices_flat, dtype=image.dtype)
    image_flat.scatter_add_(0, indices_flat, ones)

    # Reshape the image back to its original shape
    return image_flat.view_as(image)


def create_caustic(bspline, coordgrid, photon_shape=4096, caustic_shape=512, device='cuda'):

    with torch.enable_grad():
        # coords = grid(photon_shape, device).requires_grad_(True)
        coords = coordgrid.requires_grad_(True)

        # must map coords to [0,1] as this is the supported range for the spline
        # must do *.25 to decrease hf variation, must do + .75 to lift hf up a bit, to get better caustics
        hf = bspline(coords * 0.5 + 0.5).reshape(photon_shape, photon_shape) * 0.25 + 0.75   # for cat
        # hf = bspline(coords * 0.5 + 0.5).reshape(photon_shape, photon_shape) * 0.25 + 0.125  # for siggraph
        # hf = bspline(coords * 0.5 + 0.5).reshape(photon_shape, photon_shape) * 0.25 + hf_offset
        grads = torch.autograd.grad(hf.sum(), coords)[0].detach()

    # compute normals
    dx, dy = grads[..., 0], grads[..., 1]
    hf_normals = F.normalize(torch.stack((-dx, -dy, torch.ones_like(hf) * 2), dim=-1), p=2, dim=-1)

    ray_origins = torch.cat([coords.detach(), hf[..., None]], dim=-1)  # [x, y, hf]
    ray_directions = torch.tensor([0., 0., -1.], device=device).expand(photon_shape, photon_shape, 3)

    refracted_dirs = refract(ray_directions, hf_normals)

    # can save some computation here by reusing ray_dirs: ray dirs go straight down, groundplane normal goes straight up
    # plane_normals = torch.tensor([0., 0., 1.], device=device).expand(photon_shape, photon_shape, 3)
    intersections = intersect_plane(ray_origins, refracted_dirs, plane_normal=-ray_directions)

    caustic = splat(intersections, caustic_shape)
    return tonemap(caustic)


def tonemap(x):
    x /= torch.quantile(x, 0.995)
    x = x.clip(0.0, 1.0)
    return x[None, None, ...]

