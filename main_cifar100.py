import argparse
import os
import os.path
import random
import time

import numpy as np
import torch
from model import AlexNet, ResNet18
from train_util import get_model, train_model, get_results


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tstart = time.time()
    # Device Setting
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    save_path = f"./results/{args.dataset}/{args.exp}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load CIFAR100 DATASET
    if args.dataset == "cifar100-10":
        from dataloader import cifar100 as cf100

        data, taskcla, inputsize = cf100.get_10split(seed=args.seed, pc_valid=args.pc_valid)
        task_list = list(range(10))
        acc_matrix = np.zeros((10, 10))
    elif args.dataset == "cifar100-20":
        from dataloader import cifar100 as cf100

        data, taskcla, inputsize = cf100.get_20split(seed=args.seed, pc_valid=args.pc_valid)
        task_list = list(range(20))
        acc_matrix = np.zeros((20, 20))
    elif args.dataset == "miniimagenet":
        from dataloader import miniimagenet as data_loader

        dataloader = data_loader.DatasetGen(args)
        taskcla, inputsize = dataloader.taskcla, dataloader.inputsize

        task_list = list(range(20))

        acc_matrix = np.zeros((20, 20))
    else:
        raise ValueError("Invalid dataset")

    criterion = torch.nn.CrossEntropyLoss()

    if args.method == "GPM":
        if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
            from gpm import (
                get_representation_matrix_alexnet,
                test,
                train,
                train_projected,
                update_GPM,
            )
        elif args.dataset == "miniimagenet":
            from gpm import (
                get_representation_matrix_ResNet18,
                test,
                train,
                train_projected_Resnet18,
                update_GPM,
            )
        else:
            raise ValueError("Invalid dataset")

    elif args.method == "SGP":
        if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
            from gpm import (
                get_representation_matrix_alexnet,
                test,
                train,
                train_projected,
            )
            from sgp import update_SGP
        elif args.dataset == "miniimagenet":
            from gpm import (
                get_representation_matrix_ResNet18,
                test,
                train,
                train_projected_Resnet18,
            )
            from sgp import update_SGP
        else:
            raise ValueError("Invalid dataset")
    elif args.method == "GPCNS":
        if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
            from gpm import train, test
            from gpcns import train_projected, get_representation_matrix, update_GPCNS
        elif args.dataset == "miniimagenet":
            from gpm import train, test
            from gpcns import (
                train_projected_Resnet18,
                get_representation_matrix_ResNet18,
                update_GPCNS,
            )
        else:
            raise ValueError("Invalid dataset")
    else:
        raise ValueError("Invalid method {}".format(args.method))

    for task_id, ncla in taskcla:
        # specify threshold hyperparameter
        if args.method == "GPM":
            if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                threshold = np.array([0.97] * 5) + task_id * np.array([0.003] * 5)
            elif args.dataset == "miniimagenet":
                threshold = np.array([0.985] * 20) + task_id * np.array([0.0003] * 20)
                data = dataloader.get(task_id)
            else:
                raise ValueError("Invalid dataset")
        elif args.method in ["SGP", "GPCNS"]:
            if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                threshold = np.array([args.gpm_eps] * 5) + task_id * np.array([args.gpm_eps_inc] * 5)
            elif args.dataset == "miniimagenet":
                threshold = np.array([args.gpm_eps] * 20) + task_id * np.array([args.gpm_eps_inc] * 20)
                data = dataloader.get(task_id)
            else:
                raise ValueError("Invalid dataset")

        print("*" * 100)
        xtrain = data[task_id]["train"]["x"].to(device)
        ytrain = data[task_id]["train"]["y"].to(device)
        xvalid = data[task_id]["valid"]["x"].to(device)
        yvalid = data[task_id]["valid"]["y"].to(device)
        xtest = data[task_id]["test"]["x"].to(device)
        ytest = data[task_id]["test"]["y"].to(device)

        if task_id == 0:
            if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                model = AlexNet(taskcla).to(device)
            elif args.dataset == "miniimagenet":
                model = ResNet18(taskcla, 20).to(device)

            feature_list = []
            importance_list = []  # SGP, GPCNS
            # GPCNS
            Nullspace_common_list = []
            Gradient_alltask_list = []
            Nullspace_alltask_list = []
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

            fisher = train_model(
                train_fx=train,
                test_fx=test,
                args=args,
                n_epochs=args.n_epochs,
                task_id=task_id,
                xtrain=xtrain,
                ytrain=ytrain,
                xvalid=xvalid,
                yvalid=yvalid,
                model=model,
                optimizer=optimizer,
                device=device,
                criterion=criterion,
                init_lr=args.lr,
                lr_min=args.lr_min,
            )

            # Test
            test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, task_id)
            print("Test: loss={:.3f}, acc={:.2f}%".format(test_loss, test_acc))

            # Memory Update
            if args.method == "GPM":
                if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                    mat_list = get_representation_matrix_alexnet(model, device, xtrain, ytrain)
                elif args.dataset == "miniimagenet":
                    mat_list = get_representation_matrix_ResNet18(model, device, xtrain, ytrain)

                feature_list = update_GPM(model, mat_list, threshold, feature_list)

            elif args.method == "SGP":
                if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                    mat_list = get_representation_matrix_alexnet(model, device, xtrain, ytrain)
                elif args.dataset == "miniimagenet":
                    mat_list = get_representation_matrix_ResNet18(model, device, xtrain, ytrain)

                feature_list, importance_list = update_SGP(
                    args,
                    model,
                    mat_list,
                    threshold,
                    feature_list,
                    importance_list,
                )

            elif args.method == "GPCNS":
                if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                    mat_list = get_representation_matrix(model, device, xtrain, ytrain)
                elif args.dataset == "miniimagenet":
                    mat_list = get_representation_matrix_ResNet18(model, device, xtrain, ytrain)

                (
                    Nullspace_common_list,
                    importance_list,
                    Gradient_alltask_list,
                    Nullspace_alltask_list,
                ) = update_GPCNS(
                    args,
                    model,
                    mat_list,
                    device,
                    threshold,
                    Gradient_alltask_list,
                    Nullspace_common_list,
                    importance_list,
                    Nullspace_alltask_list,
                )

        else:
            if args.method == "GPM":
                feature_mat = []
                for i in range(len(feature_list)):
                    Uf = torch.Tensor(np.dot(feature_list[i], feature_list[i].transpose())).to(device)
                    feature_mat.append(Uf)
                add_on = {"feature_mat": feature_mat}

            elif args.method == "SGP":
                feature_mat = []
                for i in range(len(feature_list)):
                    Uf = torch.Tensor(
                        np.dot(
                            feature_list[i],
                            np.dot(np.diag(importance_list[i]), feature_list[i].transpose()),
                        )
                    ).to(device)
                    Uf.requires_grad = False
                    feature_mat.append(Uf)
                add_on = {"feature_mat": feature_mat}

            elif args.method == "GPCNS":
                projection_list = []
                # Projection Matrix Precomputation
                for i in range(len(Nullspace_common_list)):
                    projection_operator = torch.mm(
                        Nullspace_common_list[i],
                        torch.mm(
                            torch.diag(importance_list[i]),
                            Nullspace_common_list[i].transpose(0, 1),
                        ),
                    )
                    # Uf=torch.Tensor(np.dot(feature_list[i],np.dot(np.diag(importance_list[i]),feature_list[i].transpose()))).to(device)
                    # print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                    # Uf.requires_grad = False
                    projection_list.append(projection_operator)
                add_on = {"projection_list": projection_list}

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

            if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                train_model(
                    train_fx=train_projected,
                    test_fx=test,
                    args=args,
                    n_epochs=args.n_epochs,
                    task_id=task_id,
                    xtrain=xtrain,
                    ytrain=ytrain,
                    xvalid=xvalid,
                    yvalid=yvalid,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    criterion=criterion,
                    init_lr=args.lr,
                    lr_min=args.lr_min,
                    xtest=xtest,
                    ytest=ytest,
                    **add_on,
                )
            elif args.dataset == "miniimagenet":
                train_model(
                    train_fx=train_projected_Resnet18,
                    test_fx=test,
                    args=args,
                    n_epochs=args.n_epochs,
                    task_id=task_id,
                    xtrain=xtrain,
                    ytrain=ytrain,
                    xvalid=xvalid,
                    yvalid=yvalid,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    criterion=criterion,
                    init_lr=args.lr,
                    lr_min=args.lr_min,
                    xtest=xtest,
                    ytest=ytest,
                    **add_on,
                )

            # Test
            test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, task_id)
            print("Test: loss={:.3f}, acc={:.2f}%".format(test_loss, test_acc))

            if args.merge_list:
                print("=============== start the merge training ==============")
                old_model = get_model(model)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr2)
                fisher = train_model(
                    train_fx=train,
                    test_fx=test,
                    args=args,
                    n_epochs=args.merge_list[-1],
                    task_id=task_id,
                    xtrain=xtrain,
                    ytrain=ytrain,
                    xvalid=xvalid,
                    yvalid=yvalid,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    criterion=criterion,
                    init_lr=args.lr2,
                    lr_min=args.lr2_min,
                    merge=True,
                    old_model=old_model,
                    old_fisher=fisher,
                )

                # Test
                test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, task_id)
                print("Test: loss={:.3f}, acc={:.2f}%".format(test_loss, test_acc))

            # Memory Update
            if args.method == "GPM":
                if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                    mat_list = get_representation_matrix_alexnet(model, device, xtrain, ytrain)
                elif args.dataset == "miniimagenet":
                    mat_list = get_representation_matrix_ResNet18(model, device, xtrain, ytrain)

                feature_list = update_GPM(model, mat_list, threshold, feature_list)

            elif args.method == "SGP":
                if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                    mat_list = get_representation_matrix_alexnet(model, device, xtrain, ytrain)
                elif args.dataset == "miniimagenet":
                    mat_list = get_representation_matrix_ResNet18(model, device, xtrain, ytrain)

                feature_list, importance_list = update_SGP(
                    args,
                    model,
                    mat_list,
                    threshold,
                    feature_list,
                    importance_list,
                )

            elif args.method == "GPCNS":
                if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                    mat_list = get_representation_matrix(model, device, xtrain, ytrain)
                elif args.dataset == "miniimagenet":
                    mat_list = get_representation_matrix_ResNet18(model, device, xtrain, ytrain)

                (
                    Nullspace_common_list,
                    importance_list,
                    Gradient_alltask_list,
                    Nullspace_alltask_list,
                ) = update_GPCNS(
                    args,
                    model,
                    mat_list,
                    device,
                    threshold,
                    Gradient_alltask_list,
                    Nullspace_common_list,
                    importance_list,
                    Nullspace_alltask_list,
                )

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0 : task_id + 1]:
            xxtest = data[ii]["test"]["x"].to(device)
            yytest = data[ii]["test"]["y"].to(device)
            _, acc_matrix[task_id, jj] = test(args, model, device, xxtest, yytest, criterion, ii)
            jj += 1
        print("Accuracies =")
        for i in range(task_id + 1):
            print("\t", end="")
            for j_a in range(acc_matrix.shape[1]):
                print("{:5.2f}% ".format(acc_matrix[i, j_a]), end="")
            print()

    # Simulation Results
    get_results(args, task_list, acc_matrix, tstart, model, save_path)


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="test", help="Experiment name")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number (default: 0)")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar100-10", "cifar100-20", "miniimagenet"],
        default="cifar100-10",
        help="Dataset name",
    )
    parser.add_argument("--method", type=str, choices=["GPM", "SGP", "GPCNS"], default="GPM")
    parser.add_argument(
        "--batch_size_train",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--batch_size_test",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--pc_valid",
        default=0.05,
        type=float,
        help="fraction of training data used for validation",
    )
    # Stage 1
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of training epochs/task (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-5,
        metavar="LRM",
        help="minimum lr rate (default: 1e-5)",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=6,
        metavar="LRP",
        help="hold before decaying lr (default: 6)",
    )
    parser.add_argument(
        "--lr_factor",
        type=int,
        default=2,
        metavar="LRF",
        help="lr decay factor (default: 2)",
    )
    # SGP, GPCNS specific
    parser.add_argument(
        "--scale_coff",
        type=int,
        default=10,
        metavar="SCF",
        help="importance co-efficeint (default: 10)",
    )
    parser.add_argument(
        "--gpm_eps",
        type=float,
        default=0.97,
        metavar="EPS",
        help="threshold (default: 0.97)",
    )
    parser.add_argument(
        "--gpm_eps_inc",
        type=float,
        default=0.003,
        metavar="EPSI",
        help="threshold increment per task (default: 0.003)",
    )

    # Stage 2 (Merge)
    parser.add_argument(
        "--lr2",
        type=float,
        default=0.01,
        metavar="LR2",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lr2_min",
        type=float,
        default=1e-5,
        metavar="LR2M",
        help="minimum lr rate (default: 1e-5)",
    )
    parser.add_argument(
        "--merge_list",
        type=int,
        nargs="+",
        default=None,
        help="merge the model at the given task",
    )
    parser.add_argument("--early_stop", default=True, help="early stop for stage2 training")
    parser.add_argument("--fisher", default=True)
    parser.add_argument("--fisher_gamma", type=float, default=1.0)
    parser.add_argument(
        "--fisher_comp",
        type=str,
        choices=["true", "empirical"],
        default="true",
        help="Fisher computation, true or empirical",
    )

    args = parser.parse_args()
    print("=" * 100)
    print("Arguments =")
    for arg in vars(args):
        print("\t" + arg + ":", getattr(args, arg))
    print("=" * 100)

    main(args)
