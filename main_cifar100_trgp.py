import argparse
import os
import os.path
import random
import time

import numpy as np
import torch
from model_trgp import AlexNet, ResNet18
from train_util import get_model, train_model, get_results
from trgp import (
    get_representation_and_gradient,
    get_representation_and_gradient_Resnet18,
    get_space_and_grad,
    grad_proj_cond,
    grad_proj_cond_Resnet18,
    test,
    train,
    train_projected_regime,
    train_projected_regime_Resnet18,
    train_wo_proj,
)


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

    memory = {}
    task_name_list = []

    for task_id, ncla in taskcla:
        # specify threshold hyperparameter
        if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
            threshold = np.array([0.97] * 5) + task_id * np.array([0.003] * 5)
            layer_num = 5
        elif args.dataset == "miniimagenet":
            threshold = np.array([0.985] * 20) + task_id * np.array([0.0003] * 20)
            layer_num = 20
            data = dataloader.get(task_id)
        task_name = data[task_id]["name"]
        task_name_list.append(task_name)

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
                model = ResNet18(taskcla, 20).to(device)  # base filters: 20
            # for k_t, (m, param) in enumerate(model.named_parameters()):
            #     print(k_t, m, param.shape)
            memory[task_name] = {}

            kk = 0
            for k_t, (m, param) in enumerate(model.named_parameters()):
                if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                    if "weight" in m and "bn" not in m:
                        # print((k_t, m, param.shape))
                        memory[task_name][str(kk)] = {
                            "space_list": {},
                            "grad_list": {},
                            "regime": {},
                        }
                        kk += 1
                elif args.dataset == "miniimagenet":
                    if len(param.shape) == 4:
                        memory[task_name][str(kk)] = {
                            "space_list": {},
                            "grad_list": {},
                            "regime": {},
                        }
                        kk += 1

            space_list_all = []
            normal_param = [param for name, param in model.named_parameters() if "scale" not in name]
            optimizer = torch.optim.SGD([{"params": normal_param}], lr=args.lr)

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
            if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                mat_list, grad_list = get_representation_and_gradient(model, device, xtrain, ytrain)
            elif args.dataset == "miniimagenet":
                mat_list, grad_list = get_representation_and_gradient_Resnet18(model, device, xtrain, ytrain)
            space_list_all = get_space_and_grad(
                mat_list,
                threshold,
                memory,
                task_name,
                task_name_list,
                task_id,
                space_list_all,
            )

        else:
            memory[task_name] = {}
            kk = 0
            # print("reinit the scale for each task")
            for k_t, (m, params) in enumerate(model.named_parameters()):
                # create the saved memory
                if "weight" in m and "bn" not in m:

                    memory[task_name][str(kk)] = {
                        "space_list": {},
                        "grad_list": {},
                        "space_mat_list": {},
                        "scale1": {},
                        "scale2": {},
                        "regime": {},
                        "selected_task": {},
                    }
                    kk += 1
                # reinitialize the scale
                if "scale" in m:
                    mask = torch.eye(params.size(0), params.size(1)).to(device)
                    params.data = mask
            # print("-----------------")
            normal_param = [param for name, param in model.named_parameters() if "scale" not in name]

            scale_param = [param for name, param in model.named_parameters() if "scale" in name]
            optimizer = torch.optim.SGD(
                [
                    {"params": normal_param},
                    {"params": scale_param, "weight_decay": 0, "lr": args.lr},
                ],
                lr=args.lr,
            )

            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(space_list_all)):
                Uf = torch.Tensor(np.dot(space_list_all[i], space_list_all[i].transpose())).to(device)
                # print("Layer {} - Projection Matrix shape: {}".format(i + 1, Uf.shape))
                feature_mat.append(Uf)

            # ==1 gradient projection condition
            # print("=== excute gradient projection condition")

            # select the regime 2, which need to learn scale
            if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                grad_proj_cond(
                    model,
                    xtrain,
                    ytrain,
                    memory,
                    task_name,
                    task_id,
                    task_name_list,
                    device,
                    optimizer,
                    criterion,
                )
            elif args.dataset == "miniimagenet":
                grad_proj_cond_Resnet18(
                    model,
                    xtrain,
                    ytrain,
                    memory,
                    task_name,
                    task_id,
                    task_name_list,
                    device,
                    optimizer,
                    criterion,
                )
            space1 = [None] * layer_num
            space2 = [None] * layer_num
            for i in range(layer_num):
                for k, task_sel in enumerate(memory[task_name][str(i)]["selected_task"]):
                    if (
                        memory[task_name][str(i)]["regime"][task_sel] == "2"
                        or memory[task_name][str(i)]["regime"][task_sel] == "3"
                    ):
                        if k == 0:
                            # space1 = []
                            # for i in range(5):
                            # change the np array to torch tensor
                            space1[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]["space_list"]).to(
                                device
                            )
                        else:
                            # space2 = []
                            # for i in range(5):
                            space2[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]["space_list"]).to(
                                device
                            )
            if space1[0] is not None:
                print("space1 is not None!")
            if space2[0] is not None:
                print("space2 is not None!")

            if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                train_model(
                    train_fx=train_projected_regime,
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
                    feature_mat=feature_mat,
                    space1=space1,
                    space2=space2,
                    xtest=xtest,
                    ytest=ytest,
                )
            elif args.dataset == "miniimagenet":
                train_model(
                    train_fx=train_projected_regime_Resnet18,
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
                    feature_mat=feature_mat,
                    space1=space1,
                    space2=space2,
                    xtest=xtest,
                    ytest=ytest,
                )

            # Test
            test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, task_id, space1, space2)
            print("Test: loss={:.3f}, acc={:.2f}%".format(test_loss, test_acc))

            # merge model training
            if args.merge_list:
                print("=============== start the merge training ==============")
                old_model = get_model(model)
                optimizer = torch.optim.SGD(
                    [
                        {"params": normal_param},
                        {"params": scale_param, "weight_decay": 0, "lr": args.lr2},
                    ],
                    lr=args.lr2,
                )
                fisher = train_model(
                    train_fx=train_wo_proj,
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
                    feature_mat=feature_mat,
                    space1=space1,
                    space2=space2,
                    merge=True,
                    old_model=old_model,
                    old_fisher=fisher,
                )

                # Test
                test_loss, test_acc = test(
                    args,
                    model,
                    device,
                    xtest,
                    ytest,
                    criterion,
                    task_id,
                    space1,
                    space2,
                )
                print("Test: loss={:.3f}, acc={:.2f}%".format(test_loss, test_acc))

            # Memory Update
            if args.dataset == "cifar100-10" or args.dataset == "cifar100-20":
                mat_list, grad_list = get_representation_and_gradient(model, device, xtrain, ytrain)
            elif args.dataset == "miniimagenet":
                mat_list, grad_list = get_representation_and_gradient_Resnet18(model, device, xtrain, ytrain)
            space_list_all = get_space_and_grad(
                mat_list,
                threshold,
                memory,
                task_name,
                task_name_list,
                task_id,
                space_list_all,
            )
            # save the scale value to memory
            idx1 = 0
            idx2 = 0
            for m, params in model.named_parameters():  # layer
                if "scale1" in m:
                    memory[task_name][str(idx1)]["scale1"] = params.data
                    idx1 += 1
                if "scale2" in m:
                    memory[task_name][str(idx2)]["scale2"] = params.data
                    idx2 += 1

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0 : task_id + 1]:
            xxtest = data[ii]["test"]["x"].to(device)
            yytest = data[ii]["test"]["y"].to(device)
            # select the regime 2, which need to learn scale
            space1 = [None] * layer_num
            space2 = [None] * layer_num
            task_test = data[ii]["name"]

            if ii > 0:
                for i in range(layer_num):
                    for k, task_sel in enumerate(memory[task_test][str(i)]["selected_task"]):
                        # print(memory[task_name]['regime'][task_sel])
                        if (
                            memory[task_test][str(i)]["regime"][task_sel] == "2"
                            or memory[task_name][str(i)]["regime"][task_sel] == "3"
                        ):
                            if k == 0:
                                # space1 = []
                                # change the np array to torch tensor
                                space1[i] = torch.FloatTensor(
                                    memory[task_name_list[task_sel]][str(i)]["space_list"]
                                ).to(device)
                                idx = 0
                                for m, params in model.named_parameters():
                                    if "scale1" in m:
                                        params.data = memory[task_test][str(idx)]["scale1"].to(device)
                                        idx += 1
                            else:
                                # space2 = []
                                space2[i] = torch.FloatTensor(
                                    memory[task_name_list[task_sel]][str(i)]["space_list"]
                                ).to(device)
                                idx = 0
                                for m, params in model.named_parameters():
                                    if "scale2" in m:
                                        params.data = memory[task_test][str(idx)]["scale2"].to(device)
                                        idx += 1

            _, acc_matrix[task_id, jj] = test(
                args,
                model,
                device,
                xxtest,
                yytest,
                criterion,
                ii,
                space1=space1,
                space2=space2,
            )
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
        default="cifar100-10",
        choices=["cifar100-10", "cifar100-20", "miniimagenet"],
        help="Dataset name",
    )
    parser.add_argument("--method", type=str, default="TRGP")
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
