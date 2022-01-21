from lensiren.torch import Siren, MAML, TNGDataset
import torch


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

    # optimizer = torch.optim.Adam()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR
    maml = MAML(
        siren,
        step_size=args.step_size,
        learn_step_size=args.learn_step_size,
        per_param_step_size=args.per_param_step_size,
        num_adaptation_steps=args.num_adaptation_steps,
        loss_function=torch.nn.MSELoss(),
        device=torch.cuda.device(0)
    )
    pass


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", name="hkappa188hst_TNG100_rau_spl.h5")
    parser.add_argument("--first_omega", default=30., type=float)
    parser.add_argument("--hidden_omega", default=30., type=float)
    parser.add_argument("--per_param_step_size", action="store_true")
    parser.add_argument("--step_size", default=1e-3, type=float)
    parser.add_argument("--learn_step_size", action="store_true")
    parser.add_argument("--per_param_step_size", action="store_true")
    parser.add_argument("--num_adaptaion_steps", default=1, type=int, help="inner optimization steps")
    args = parser.parse_args()
    main(args)