import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import trange

from utils import BOUNDS, ForwardProcess, NoisePredictor, ReverseProcess, normalise, plot_samples


def run(args):
    # Load landmarks of first frame in sequence
    X = np.load("landmarks.npy")[0] 
    X = np.repeat(X, args.dataset_size, axis=0)
    print("landmarks shape", X.shape)

    X = torch.Tensor(X)

    # Standardise data
    if args.normalise:
        # scale inputs between -1 and 1
        X = normalise(X)
    else:
        X -= X.mean(dim=0, keepdim=True)
        X /= X.std(dim=0, keepdim=True)

    # Invert y-axis so image is right-side up
    X[:, 1] = -X[:, 1]

    # plot the landmarks
    if args.debug:
        fig, ax = plt.subplots(figsize=(15, 15))
        plot_samples(X, ax=ax)
        plt.show()

    T = args.num_steps  # no. diffusion steps
    betas = torch.linspace(0.0, 0.99, T + 1)
    fp = ForwardProcess(betas=betas)

    if args.debug:
        _, ax = plt.subplots(1, 3, figsize=(15, 5))

        # add noise to X (landmarks) at t=[0, 2, T] timesteps and plot the results
        for idx, t in enumerate([0, 2, T]):
            x_t = fp.get_x_t(X, t=torch.LongTensor([t]))[1].numpy()  # returns the perturbed sample
            plot_samples(x_t, ax=ax[idx])
            ax[idx].set(xlim=BOUNDS, ylim=BOUNDS, title=f"t={t}", aspect="equal")
        plt.show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')

    model = NoisePredictor(
        T=T,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_diffusion_blocks=args.num_diffusion_blocks
    ).to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    N = X.shape[0]
    for _epoch in trange(args.num_epochs):

        with torch.no_grad():
            # Sample N random t's
            t = torch.randint(low=1, high=T + 1, size=(N,))
            t = torch.LongTensor(t)

            # Get the noise added and the noisy version of the data using the forward
            # process given t
            eps_0, x_t = fp.get_x_t(X, t=t)

        # Predict the noise added to x_0 from x_t
        pred_eps = model(x_t.to(device), t.to(device))

        # Simplified objective without weighting with alpha terms (Ho et al, 2020)
        loss = torch.nn.functional.mse_loss(pred_eps, eps_0.to(device))
        if args.debug:
            print(f'Epoch: {_epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    rp = ReverseProcess(betas=betas, model=model, device=device)
    samples = rp.sample(n_samples=args.num_test_samples).cpu().numpy()

    fig, ax = plt.subplots(figsize=(15, 15))
    plot_samples(samples, ax=ax)
    if args.save_fig:
        plt.savefig('submission.png', bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--num_test_samples', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_diffusion_blocks', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=1)
    parser.add_argument('--normalise', action='store_true')
    parser.add_argument('--save_fig', action='store_true')
    parser.add_argument('--debug', action='store_true')

    run(parser.parse_args())

