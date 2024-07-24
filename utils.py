import torch
from torch import nn
import seaborn as sns

BOUNDS = (-2.5, 2.5)


def normalise(x):
    # scale between [-1, 1]
    x_scaled = (x - x.min()) / (x.max() - x.min()) * 2 - 1
    
    return x_scaled


def plot_samples(samples, ax):
    sns.scatterplot(
        x=samples[:, 0],
        y=samples[:, 1],
        ax=ax,
        s=10,
        alpha=0.8,
        edgecolor=None,
        linewidth=0,
    )
    ax.set(xlim=BOUNDS, ylim=BOUNDS, aspect="equal")


class ForwardProcess:
    def __init__(self, betas: torch.Tensor):
        self.beta = betas

        """
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=-1)
        """

        # cosine-beta schedule
        # taken from: https://github.com/albarji/toy-diffusion/blob/master/swissRoll.ipynb
        s = 0.008
        diffusion_steps = len(betas)
        timesteps = torch.tensor(range(0, diffusion_steps), dtype=torch.float32)
        schedule = torch.cos((timesteps / diffusion_steps + s) / (1 + s) * torch.pi / 2)**2

        self.alpha_bar = schedule / schedule[0]
        self.betas = 1 - self.alpha_bar / torch.concatenate([self.alpha_bar[0:1], self.alpha_bar[0:-1]])
        self.alphas = 1 - betas

    def get_x_t(
        self, x_0: torch.Tensor, t: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process given the unperturbed sample x_0.

        Args:
            x_0: Original, unperturbed samples.
            t: Target timestamp of the diffusion process of each sample.

        Returns:
            Noise added to original sample and perturbed sample.
        """
        eps_0 = torch.randn_like(x_0).to(x_0)  # N, 2 -> random noise
        alpha_bar = self.alpha_bar[t, None]  # N, 1
        mean = (alpha_bar**0.5) * x_0  # N, 2
        std = (1.0 - alpha_bar) ** 0.5  # N, 1

        return eps_0, mean + std * eps_0


class ReverseProcess(ForwardProcess):
    def __init__(self, betas: torch.Tensor, model: nn.Module, device: torch.device):
        super().__init__(betas)
        self.model = model
        self.device = device
        self.T = len(betas) - 1

        self.sigma = (
            (1 - self.alphas)
            * (1 - torch.roll(self.alpha_bar, 1))
            / (1 - self.alpha_bar)
        ) ** 0.5
        self.sigma[1] = 0.0

    def get_x_t_minus_one(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            t_vector = torch.full(size=(len(x_t),), fill_value=t, dtype=torch.long)  # vector filled with t values
            eps = self.model(x_t.to(self.device), t=t_vector.to(self.device)).cpu()  # predicted noise

        eps *= (1 - self.alphas[t]) / ((1 - self.alpha_bar[t]) ** 0.5)
        mean = 1 / (self.alphas[t] ** 0.5) * (x_t - eps)
       
        return mean + self.sigma[t] * torch.randn_like(x_t)

    def sample(self, n_samples):
        # Initialize with X_T ~ N(0, I)
        x_t = torch.randn(n_samples, 2)  # reverse diffusion, init noise
        
        # reverse diffusion process
        for t in range(self.T, 0, -1):
            x_t = self.get_x_t_minus_one(x_t, t=t)

        return x_t


class DiffusionBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)

        return self.activation(x)


class NoisePredictor(nn.Module):

    def __init__(self, T, embedding_dim=1, hidden_dim=200, num_diffusion_blocks=3):
        super().__init__()
        self.T = T
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_blocks = num_diffusion_blocks
        self.input_dim = 2 + embedding_dim

        self.blocks = [DiffusionBlock(dim=self.hidden_dim) for _ in range(self.num_diffusion_blocks)]

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(100, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            
            *self.blocks,
            
            nn.Linear(self.hidden_dim, 100),
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(100, 20),
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(20, 2)
        )

        self.embedding = nn.Embedding(self.T + 1, self.embedding_dim)

    def forward(self, x_t, t):
        t_embedding = self.embedding(t)

        inp = torch.cat([x_t, t_embedding], dim=1)

        return self.model(inp)
