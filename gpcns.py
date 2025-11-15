import numpy as np
import torch


def train_projected(
    args,
    model,
    device,
    x,
    y,
    optimizer,
    criterion,
    task_id,
    projection_list,
):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r):
            b = r[i : i + args.batch_size_train]
        else:
            b = r[i:]
        b = b.cpu()
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[task_id], target)
        loss.backward()
        # Gradient Projections

        conv_idx = 0
        for k, (m, params) in enumerate(model.named_parameters()):

            if conv_idx < 5:
                update = params.grad.data

                if len(update.shape) == 4:
                    update_size = update.size(0)
                    params.grad.data = params.grad.data - torch.mm(
                        params.grad.data.view(update_size, -1),
                        projection_list[conv_idx],
                    ).view(params.size())

                    conv_idx += 1

                elif len(update.shape) == 2:
                    params.grad.data = params.grad.data - torch.mm(
                        params.grad.data, projection_list[conv_idx]
                    )
                    conv_idx += 1
            if (k < 15 and len(params.size()) == 1) and task_id != 0:
                params.grad.data.fill_(0)

        optimizer.step()


def train_projected_Resnet18(
    args,
    model,
    device,
    x,
    y,
    optimizer,
    criterion,
    task_id,
    projection_list,
):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r):
            b = r[i : i + args.batch_size_train]
        else:
            b = r[i:]
        b = b.cpu()
        data = x[b]
        # print(i)

        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[task_id], target)
        loss.backward()

        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if len(params.size()) == 4 and "weight" in m:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(
                    params.grad.data.view(sz, -1), projection_list[kk]
                ).view(params.size())

                kk += 1
            elif len(params.size()) == 1 and task_id != 0:
                params.grad.data.fill_(0)

        optimizer.step()


def get_representation_matrix(net, device, x, y=None):
    conv_idx = 0
    mat_list = []
    for k, (m, params) in enumerate(net.named_parameters()):
        if conv_idx < 5:
            update = params.grad.data
            update_size = update.shape[0]
            if len(update.shape) == 4:
                mat_list.append(params.grad.data.view(update_size, -1))
                conv_idx += 1

            elif len(update.shape) == 2:
                mat_list.append(update)
                conv_idx += 1
        else:
            break
    return mat_list


def get_representation_matrix_ResNet18(net, device, x, y=None):
    # Collect activations by forward pass
    grad_list = []

    k_conv = 0
    for k, (m, params) in enumerate(net.named_parameters()):
        if len(params.shape) == 4 and "weight" in m:

            grad = params.grad.data
            grad = grad.reshape(
                grad.shape[0], grad.shape[1] * grad.shape[2] * grad.shape[3]
            )
            grad_list.append(grad)
            k_conv += 1
    return grad_list


def update_GPCNS(
    args,
    model,
    mat_list,
    device,
    threshold_list,
    Gradient_alltask_list,
    Nullspace_common_list,
    importance_list,
    Nullspace_alltask_list,
):

    print("threshold_list: ", threshold_list)

    if not Nullspace_common_list:
        # After First Task
        for i in range(len(mat_list)):
            Gradient_matrix = mat_list[i]
            Gradient_alltask_list.append(Gradient_matrix)

            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(
                Gradient_alltask_list[i], full_matrices=False
            )
            V_matrix = Vh_matrix.transpose(0, 1)

            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [
                torch.sum(Sigma[0 : idx + 1]) for idx in range(len(Sigma))
            ]
            Sigma_select = (
                torch.tensor(Sigma_accumul_list).to(device)
                <= threshold_list[i] * Sigma_total
            )
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)

            V_tilde_matrix = V_matrix[:, :rank_subspace]

            Nullspace_common_list.append(V_tilde_matrix)
            Nullspace_alltask_list.append(V_tilde_matrix)

            importance_list.append(
                torch.ones(Nullspace_common_list[i].shape[1]).to(device)
            )

    else:
        for i in range(len(mat_list)):
            Gradient_matrix = mat_list[i]
            Gradient_alltask_list[i] = torch.vstack(
                [Gradient_alltask_list[i], Gradient_matrix]
            )

            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(
                Gradient_alltask_list[i], full_matrices=False
            )
            V_matrix = Vh_matrix.transpose(0, 1)

            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [
                torch.sum(Sigma[0 : idx + 1]).item() for idx in range(len(Sigma))
            ]
            Sigma_select = (
                torch.tensor(Sigma_accumul_list).to(device)
                <= threshold_list[i] * Sigma_total
            )
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)

            V_tilde_matrix = V_matrix[:, :rank_subspace]

            Nullspace_alltask_list[i] = torch.hstack(
                [Nullspace_alltask_list[i], V_tilde_matrix]
            )
            U_matrix, Sigma, Vh_matrix = torch.linalg.svd(
                Nullspace_alltask_list[i], full_matrices=False
            )

            Sigma_total = torch.sum(Sigma)
            Sigma_accumul_list = [
                torch.sum(Sigma[0 : idx + 1]).item() for idx in range(len(Sigma))
            ]
            Sigma_select = (
                torch.tensor(Sigma_accumul_list).to(device)
                <= threshold_list[i] * Sigma_total
            )
            Sigma_slim = Sigma[Sigma_select]
            rank_subspace = torch.count_nonzero(Sigma_slim)

            Nullspace_common_list[i] = U_matrix[:, :rank_subspace]

            importance = ((args.scale_coff + 1) * Sigma[:rank_subspace]) / (
                args.scale_coff * Sigma[:rank_subspace] + max(Sigma[:rank_subspace])
            )

            importance_list[i] = importance

    # print("-" * 40)
    # print("Common Null Space Matrix")
    # print("-" * 40)
    # for i in range(len(Nullspace_common_list)):
    #     print("Layer {} : {} * {}".format(i + 1, Nullspace_common_list[i].shape[0], Nullspace_common_list[i].shape[1]))
    # print("-" * 40)
    return (
        Nullspace_common_list,
        importance_list,
        Gradient_alltask_list,
        Nullspace_alltask_list,
    )
