import torch
from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        super().__init__()
        self.register_buffer(
            "frequencies",
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        # Sin/cos positional encoding for 3D coordinates.
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)
        embedding_dim = n_harmonic_functions * 2 * 3

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
        )
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
        )
        self.density_layer[0].bias.data[0] = -1.5

    def _get_densities(self, features):
        # Map MLP features to volumetric density values.
        raw = self.density_layer(features)
        return 1 - (-raw).exp()

    def _get_colors(self, features, rays_directions):
        # Predict colors conditioned on view direction embeddings.
        spatial_size = features.shape[:-1]
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)
        rays_embedding = self.harmonic_embedding(rays_directions_normed)
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )
        color_input = torch.cat((features, rays_embedding_expand), dim=-1)
        return self.color_layer(color_input)

    def forward(self, ray_bundle: RayBundle, **kwargs):
        # Full-field forward pass for a batch of rays.
        points_world = ray_bundle_to_ray_points(ray_bundle)
        embeds = self.harmonic_embedding(points_world)
        features = self.mlp(embeds)
        densities = self._get_densities(features)
        colors = self._get_colors(features, ray_bundle.directions)
        return densities, colors

    def batched_forward(self, ray_bundle: RayBundle, n_batches: int = 16, **kwargs):
        # Memory-friendly forward pass that splits rays into batches.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples, device=ray_bundle.origins.device), n_batches)
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            )
            for batch_idx in batches
        ]
        rays_densities, rays_colors = [
            torch.cat([batch_output[output_i] for batch_output in batch_outputs], dim=0)
            .view(*spatial_size, -1)
            for output_i in (0, 1)
        ]
        return rays_densities, rays_colors


def make_implicit_renderer(image_height, image_width, n_pts_per_ray, min_depth, max_depth, n_rays_per_image):
    # Build Monte Carlo sampler and full raster renderers for training/previews.
    raysampler_mc = MonteCarloRaysampler(
        min_x=-1,
        max_x=1,
        min_y=-1,
        max_y=1,
        n_rays_per_image=n_rays_per_image,
        n_pts_per_ray=n_pts_per_ray,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    raysampler_full = NDCMultinomialRaysampler(
        image_width=image_width,
        image_height=image_height,
        n_pts_per_ray=n_pts_per_ray,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    raymarcher = EmissionAbsorptionRaymarcher()
    renderer_mc = ImplicitRenderer(raysampler_mc, raymarcher)
    renderer_full = ImplicitRenderer(raysampler_full, raymarcher)
    renderer_full.render_cfg = {
        "image_height": image_height,
        "image_width": image_width,
    }
    return renderer_mc, renderer_full
