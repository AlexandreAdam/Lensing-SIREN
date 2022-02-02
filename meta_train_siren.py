from lensiren.torch import Siren, MAML, TNGDataset
import torch
import torch.nn.functional as F
from torchmeta.utils.data import MetaDataLoader
import os


def main(args):
    siren = Siren(
        in_features=10,
        hidden_features=10,
        hidden_layers=args.hidden_layers,
        out_features=1,
        outermost_linear=False,
        first_omega_0=args.first_omega,
        hidden_omega_0=args.hidden_omega
    )

    optim = torch.optim.Adam(siren.parameters(), lr=args.learning_rate)
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    maml = MAML(
        siren,
        step_size=args.step_size,
        learn_step_size=args.learn_step_size,
        per_param_step_size=args.per_param_step_size,
        num_adaptation_steps=args.num_adaptation_steps,
        loss_function=F.mse_loss,
        optimizer=optim,
        device=device
    )

    filepath = os.path.join(os.getenv("LSIREN_PATH"), "data", args.dataset)
    dataloader = MetaDataLoader(TNGDataset(filepath, meta_train=True), batch_size=args.batch_size)
    maml.train(dataloader, max_batches=args.max_batches)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="hkappa188hst_TNG100_rau_spl.h5")
    parser.add_argument("--first_omega", default=30., type=float)
    parser.add_argument("--hidden_omega", default=30., type=float)
    parser.add_argument("--hidden_layers", default=2, type=int, help="Number of SIREN hidden layers")
    parser.add_argument("--learning_rate", default=1e-3, help="Outer loop learning rate")
    parser.add_argument("--step_size", default=1e-3, type=float, help="Inner loop learning rate")
    parser.add_argument("--learn_step_size", action="store_true")
    parser.add_argument("--per_param_step_size", action="store_true")
    parser.add_argument("--num_adaptation_steps", default=1, type=int, help="inner optimization steps")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of tasks in the inner loop")
    parser.add_argument("--max_batches", default=10, type=int, help="Number of batch in an epoch")
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()
    main(args)
