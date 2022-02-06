from lensiren.torch import Siren, TNGDataset, MAML
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os, json, time

DATE = datetime.now().strftime("%y%m%d%H%M%S")


def main(args):

    # Take care of where to write stuff
    model_dir = os.path.join(os.getenv("LSIREN_PATH"), "models", args.model_id)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    with open(os.path.join(model_dir, "script_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    log_dir = os.path.join(os.getenv("LSIREN_PATH"), args.log_dir, args.model_id)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

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

    meta_siren = MAML(
        num_meta_steps=args.num_adaptation_steps,
        hypo_module=siren,
        init_lr=args.step_size,
        lr_type=args.lr_type,
        loss_type="image",
        first_order=True
    )

    filepath = os.path.join(os.getenv("LSIREN_PATH"), "data", args.dataset)
    dataset = TNGDataset(filepath, indices=None if args.data_len is None else list(range(args.data_len)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    global_start = time.time()
    history = {"loss": []}
    global_step = 0
    estimated_time_for_epoch = 0
    out_of_time = False
    for epoch in range(args.epochs):
        if (time.time() - global_start) > args.max_time*3600 - estimated_time_for_epoch:
            break
        epoch_start = time.time()
        epoch_loss = 0.
        for step, sample in enumerate(dataloader):
            global_step += 1
            model_output = meta_siren(sample)
            loss = meta_siren.loss_function(sample[0], model_output["model_out"], sample[1])
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.detach().cpu().numpy()
            writer.add_scalar("Loss", loss, global_step=global_step)

        # Logs
        history["loss"].append(epoch_loss/(step + 1))
        print("Epoch %d, Total loss %0.6f" % (epoch, epoch_loss/(step + 1)))
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
        writer.add_figure(tag="Reconstruction", figure=fig, global_step=global_step)
        if args.verbose:
            plt.show()
        if not epoch % args.epochs_til_checkpoint and epoch:
            torch.save(siren.state_dict(), os.path.join(model_dir, 'model_epoch_%04d.pth' % epoch))
        if epoch > 0:  # First epoch is always very slow and not a good estimate of an epoch time.
            estimated_time_for_epoch = time.time() - epoch_start
    torch.save(siren.state_dict(), os.path.join(model_dir, 'model_final.pth'))
    return history


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="maml_siren" + "_" + DATE)
    parser.add_argument("--dataset", default="hkappa188hst_TNG100_rau_spl.h5")
    parser.add_argument("--data_len", default=2, type=int)
    parser.add_argument("--epochs",  default=2, type=int)
    parser.add_argument("--first_omega", default=0.5, type=float)
    parser.add_argument("--hidden_omega", default=0.5, type=float)
    parser.add_argument("--hidden_layers", default=2, type=int, help="Number of SIREN hidden layers")
    parser.add_argument("--hidden_features", default=50, type=int, help="Number of SIREN hidden feature per layers")
    parser.add_argument("--learning_rate", default=5e-3, help="Outer loop learning rate")
    parser.add_argument("--step_size", default=0.05, type=float, help="Inner loop learning rate")
    parser.add_argument("--loss_type", default="image", help="'image' (default), 'gradient', 'laplace', 'image_gradient', 'image_laplace', 'gradient_laplace' or 'image_gradient_laplace'")
    parser.add_argument("--lr_type", default="static", help="'static' (default), 'global', 'per_step', 'per_parameter', 'per_parameter_per_step'")
    parser.add_argument("--num_adaptation_steps", default=5, type=int, help="inner optimization steps")
    parser.add_argument("--batch_size", default=2, type=int, help="Number of tasks in the inner loop")
    parser.add_argument("--epochs_til_checkpoint", default=2)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--max_time", default=np.inf, type=float)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
