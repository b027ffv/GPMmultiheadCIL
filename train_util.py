import os
import time
from copy import deepcopy

import numpy as np
import torch
from fisher import compute_fisher_matrix_diag, compute_fisher_merging, get_avg_fisher
from tqdm import tqdm


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_model(
    train_fx,
    test_fx,
    args,
    n_epochs,
    task_id,
    xtrain,
    ytrain,
    xvalid,
    yvalid,
    model,
    optimizer,
    device,
    criterion,
    init_lr,
    lr_min,
    xtest=None,
    ytest=None,
    merge=False,
    old_model=None,
    old_fisher=None,
    **kwargs,
):
    if merge:
        assert args.merge_list is not None
        assert old_model is not None

    best_loss = np.inf
    early_stop = False
    lr = init_lr

    save_path = f"./results/{args.dataset}/{args.exp}/"
    log_file = os.path.join(save_path, "log.txt")

    loop = tqdm(range(1, n_epochs + 1))
    for epoch in loop:
        # Train
        train_fx(
            args,
            model,
            device,
            xtrain,
            ytrain,
            optimizer,
            criterion,
            task_id,
            **kwargs,
        )

        # Compute train loss
        tr_loss, tr_acc = test_fx(args, model, device, xtrain, ytrain, criterion, task_id, **kwargs)
        # Validate
        valid_loss, valid_acc = test_fx(args, model, device, xvalid, yvalid, criterion, task_id, **kwargs)

        # Adapt lr
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = get_model(model)
            patience = args.lr_patience
        else:
            patience -= 1
            if patience <= 0:
                lr /= args.lr_factor
                if merge and not args.early_stop:  # only in stage2
                    early_stop = False
                else:
                    if lr < lr_min:
                        early_stop = True
                patience = args.lr_patience
                adjust_learning_rate(optimizer, lr)

        loop.set_description(f"Task {task_id} | Epoch [{epoch}/{n_epochs}]")
        loop.set_postfix(
            t_loss="{:.3f}".format(tr_loss),
            t_acc="{:.2f}".format(tr_acc),
            v_loss="{:.3f}".format(valid_loss),
            v_acc="{:.2f}".format(valid_acc),
            lr="{:.1e}".format(lr),
        )

        # Analysis
        if epoch == 1 and xtest is not None and ytest is not None:
            test_loss, test_acc = test_fx(args, model, device, xtest, ytest, criterion, task_id, **kwargs)
            with open(log_file, "a") as f:
                f.write(
                    "Test at 1st epoch of task {}, loss: {:.3f}, acc: {:.2f}\n".format(task_id, test_loss, test_acc)
                )

        if merge and (epoch in args.merge_list or early_stop):
            cur_model = model.state_dict()
            new_model = old_model
            if args.fisher:
                # Compute current fisher
                cur_fisher = compute_fisher_matrix_diag(
                    args, model, device, optimizer, xtrain, ytrain, task_id, **kwargs
                )
                # Compute merging coefficient
                theta = compute_fisher_merging(model, old_model, cur_fisher, old_fisher)

            else:
                theta = 1 / (task_id + 1)

            for m, param in model.named_parameters():
                new_model[m] = (1 - theta) * old_model[m] + theta * cur_model[m]
            model.load_state_dict(new_model)

            # Update fisher
            if args.fisher:
                cur_fisher = compute_fisher_matrix_diag(
                    args, model, device, optimizer, xtrain, ytrain, task_id, **kwargs
                )
                print(">>merged_fisher: ", get_avg_fisher(cur_fisher))
                fisher = deepcopy(cur_fisher)
                for n in fisher.keys():
                    fisher[n] += old_fisher[n] * args.fisher_gamma
                print(">>update_fisher: ", get_avg_fisher(fisher))

        if early_stop:
            break

    if not merge:
        # Stage1
        set_model_(model, best_model)

        if task_id == 0 and args.merge_list and args.fisher:  # only for the first task
            fisher = compute_fisher_matrix_diag(args, model, device, optimizer, xtrain, ytrain, task_id, **kwargs)
            print(">>init_fisher: ", get_avg_fisher(fisher))
            return fisher
        else:
            return None
    else:
        if args.fisher:
            return fisher
        else:
            return None


def get_results(args, task_list, acc_matrix, tstart, model, save_path):
    task_order_str = "Task Order : {}".format(np.array(task_list))
    acc = acc_matrix[-1].mean()
    final_avg_accuracy_str = "Final Avg Accuracy: {:5.2f}%".format(acc)
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    backward_transfer_str = "Backward transfer: {:5.2f}%".format(bwt)
    elapsed_time_str = "[Elapsed time = {:.1f} ms]".format((time.time() - tstart) * 1000)

    print(task_order_str)
    print(final_avg_accuracy_str)
    print(backward_transfer_str)
    print(elapsed_time_str)
    print("-" * 40)

    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    log_file = os.path.join(save_path, "log.txt")

    with open(log_file, "a") as f:
        f.write("Accuracies =\n")
        for i in range(acc_matrix.shape[0]):
            f.write("\t")
            for j in range(acc_matrix.shape[1]):
                f.write("{:5.2f}% ".format(acc_matrix[i, j]))
            f.write("\n")
        f.write("-" * 60 + "\n")
        f.write(task_order_str + "\n")
        f.write(final_avg_accuracy_str + "\n")
        f.write(backward_transfer_str + "\n")
        f.write(elapsed_time_str + "\n")
        f.write("-" * 60 + "\n")
