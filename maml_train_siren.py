from lensiren.torch import Siren, TNGDataset, MAMLSiren
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt


def main(args):
    siren = Siren(
        in_features=2,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=1,
        outermost_linear=True,
        first_omega_0=args.first_omega,
        hidden_omega_0=args.hidden_omega
    )

    optim = torch.optim.Adam(siren.parameters(), lr=args.learning_rate)
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    siren.to(device)

    meta_siren = MAMLSiren(
        num_meta_steps=args.num_adaptation_steps,
        hypo_module=siren,
        init_lr=args.step_size,
        lr_type='global',
        loss_type="image",
        first_order=True
    )

    filepath = os.path.join(os.getenv("LSIREN_PATH"), "data", args.dataset)
    dataset = TNGDataset(filepath, indices=None if args.data_len is None else list(range(args.data_len)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    for epoch in range(args.epochs):
        for step, sample in enumerate(dataloader):
            model_output = meta_siren(sample)
            loss = meta_siren.loss_function(sample[0], model_output["model_out"], sample[1])

            optim.zero_grad()
            loss.backward()
            optim.step()

            if args.verbose:
                if not epoch % 10:
                    print("Epoch %d, Step %d, Total loss %0.6f" % (epoch, step, loss))
                    fig, axs = plt.subplots(args.batch_size, args.num_adaptation_steps+2, figsize=(30, 4*args.batch_size))
                    for i in range(args.batch_size):
                        coords, targets = dataset[i]
                        vmax = targets["image"].numpy().max()
                        vmin = targets["image"].numpy().min()
                        axs[i, -2].imshow(model_output["model_out"][i].view(188, 188).detach().cpu().numpy(), vmin=vmin, vmax=vmax)
                        axs[i, -2].set_title(f"Inner step {args.num_adaptation_steps}")
                        axs[i, -2].axis("off")
                        axs[i, -1].imshow(targets["image"].view(188, 188).detach().cpu().numpy())
                        axs[i, -1].set_title("Ground Truth")
                        axs[i, -1].axis("off")
                        for step in range(args.num_adaptation_steps):
                            axs[i, step].imshow(model_output["intermed_predictions"][step][i].view(188, 188).detach().cpu().numpy(), vmin=vmin, vmax=vmax)
                            axs[i, step].axis("off")
                            axs[i, step].set_title(f"Inner step {step}" if step > 0 else "Learned Initialization")
                    plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="hkappa188hst_TNG100_rau_spl.h5")
    parser.add_argument("--data_len", default=2, type=int)
    parser.add_argument("--epochs",  default=500, type=int)
    parser.add_argument("--first_omega", default=0.5, type=float)
    parser.add_argument("--hidden_omega", default=0.5, type=float)
    parser.add_argument("--hidden_layers", default=2, type=int, help="Number of SIREN hidden layers")
    parser.add_argument("--hidden_features", default=50, type=int, help="Number of SIREN hidden feature per layers")
    parser.add_argument("--learning_rate", default=5e-3, help="Outer loop learning rate")
    parser.add_argument("--step_size", default=0.05, type=float, help="Inner loop learning rate")
    parser.add_argument("--loss_type", default="image", help="'image', 'gradient', 'laplace', 'image_gradient', 'image_laplace', 'gradient_laplace' or 'image_gradient_laplace'")
    parser.add_argument("--learn_step_size", action="store_true")
    parser.add_argument("--per_param_step_size", action="store_true")
    parser.add_argument("--num_adaptation_steps", default=5, type=int, help="inner optimization steps")
    parser.add_argument("--batch_size", default=2, type=int, help="Number of tasks in the inner loop")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
