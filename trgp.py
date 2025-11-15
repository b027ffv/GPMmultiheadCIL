import numpy as np
import torch
import torch.nn.functional as F
from model import compute_conv_output_size

Eplison_1 = 0.5


def train(
    args,
    model,
    device,
    x,
    y,
    optimizer,
    criterion,
    task_id,
    space1=[None, None, None],
    space2=[None, None, None],
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
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, space1=space1, space2=space2)
        loss = criterion(output[task_id], target)
        loss.backward()
        optimizer.step()


# train with regime but without projection
def train_wo_proj(
    args,
    model,
    device,
    x,
    y,
    optimizer,
    criterion,
    task_id,
    feature_mat,
    space1=[None, None, None],
    space2=[None, None, None],
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
        output = model(data, space1=space1, space2=space2)
        loss = criterion(output[task_id], target)

        loss.backward()

        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if "weight" in m:
                if k < 21 and len(params.size()) != 1:
                    # sz =  params.grad.data.size(0)
                    # params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                    #                                         feature_mat[kk]).view(params.size())
                    kk += 1
                    continue
                elif (k < 1 and len(params.size()) == 1) and task_id != 0:
                    params.grad.data.fill_(0)
        optimizer.step()


def train_projected_regime(
    args,
    model,
    device,
    x,
    y,
    optimizer,
    criterion,
    task_id,
    feature_mat,
    space1=[None, None, None],
    space2=[None, None, None],
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
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, space1=space1, space2=space2)
        loss = criterion(output[task_id], target)
        loss.backward()

        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if "weight" in m:
                if k < 21 and len(params.size()) != 1:
                    sz = params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), feature_mat[kk]).view(
                        params.size()
                    )

                    kk += 1
                elif (k < 1 and len(params.size()) == 1) and task_id != 0:
                    params.grad.data.fill_(0)

        optimizer.step()

def train_projected_regime_Resnet18(
    args,
    model,
    device,
    x,
    y,
    optimizer,
    criterion,
    task_id,
    feature_mat,
    space1=[None, None, None],
    space2=[None, None, None],
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
        data = x[b]
        # print(i)

        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, space1=space1, space2=space2)
        loss = criterion(output[task_id], target)
        loss.backward()

        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if len(params.size()) == 4 and "weight" in m:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(
                    params.grad.data.view(sz, -1), feature_mat[kk]
                ).view(params.size())

                kk += 1
            elif len(params.size()) == 1 and task_id != 0:
                params.grad.data.fill_(0)

        optimizer.step()

def test(
    args,
    model,
    device,
    x,
    y,
    criterion,
    task_id,
    space1=[None, None, None],
    space2=[None, None, None],
    **kwargs,
):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.size(0))
    # np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0, len(r), args.batch_size_test):
            if i + args.batch_size_test <= len(r):
                b = r[i : i + args.batch_size_test]
            else:
                b = r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data, space1=space1, space2=space2)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item() * len(b)
            total_num += len(b)

    acc = 100.0 * correct / total_num
    final_loss = total_loss / total_num

    return final_loss, acc


def get_representation_and_gradient(net, device, x, y=None):
    """
    aim to get the representation (activation) and gradient(optimal) of each layer
    """

    # Collect activations by forward pass
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[0:125]  # Take 125 random samples
    example_data = x[b]
    example_data = example_data.to(device)

    batch_list = [2 * 12, 100, 100, 125, 125]
    mat_list = []
    grad_list = []  # list contains gradient of each layer
    act_key = list(net.act.keys())

    net.eval()
    example_out = net(example_data)

    for i in range(len(net.map)):
        bsz = batch_list[i]
        k = 0
        if i < 3:
            ksz = net.ksize[i]
            s = compute_conv_output_size(net.map[i], net.ksize[i])
            # logging.info("s:{}".format(s))
            mat = np.zeros((net.ksize[i] * net.ksize[i] * net.in_channel[i], s * s * bsz))
            act = net.act[act_key[i]].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:, k] = act[kk, :, ii : ksz + ii, jj : ksz + jj].reshape(-1)
                        k += 1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    # print_log("-" * 30, log)
    # print_log("Representation Matrix", log)
    # print_log("-" * 30, log)
    # for i in range(len(mat_list)):
    #     print_log("Layer {} : {}".format(i + 1, mat_list[i].shape), log)
    # print_log("-" * 30, log)
    return mat_list, grad_list

def get_representation_and_gradient_Resnet18(net, device, x, y=None):
    # Collect activations by forward pass

    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[0:100]  # ns=100 examples
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)
    grad_list = []  # list contains gradient of each layer

    net.eval()
    example_out = net(example_data)

    act_list = []
    act_list.extend(
        [
            net.act["conv_in"],
            net.layer1[0].act["conv_0"],
            net.layer1[0].act["conv_1"],
            net.layer1[1].act["conv_0"],
            net.layer1[1].act["conv_1"],
            net.layer2[0].act["conv_0"],
            net.layer2[0].act["conv_1"],
            net.layer2[1].act["conv_0"],
            net.layer2[1].act["conv_1"],
            net.layer3[0].act["conv_0"],
            net.layer3[0].act["conv_1"],
            net.layer3[1].act["conv_0"],
            net.layer3[1].act["conv_1"],
            net.layer4[0].act["conv_0"],
            net.layer4[0].act["conv_1"],
            net.layer4[1].act["conv_0"],
            net.layer4[1].act["conv_1"],
        ]
    )

    batch_list = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100,]  # scaled
    # network arch

    stride_list = [2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
    map_list = [84, 42, 42, 42, 42, 42, 21, 21, 21, 21, 11, 11, 11, 11, 6, 6, 6]
    in_channel = [3, 20, 20, 20, 20, 20, 40, 40, 40, 40, 80, 80, 80, 80, 160, 160, 160]
    pad = 1
    sc_list = [5, 9, 13]
    p1d = (1, 1, 1, 1)
    mat_final = []  # list containing GPM Matrices
    mat_list = []
    mat_sc_list = []
    for i in range(len(stride_list)):
        if i == 0:
            ksz = 3
        else:
            ksz = 3
        bsz = batch_list[i]
        st = stride_list[i]
        k = 0
        s = compute_conv_output_size(map_list[i], ksz, stride_list[i], pad)
        mat = np.zeros((ksz * ksz * in_channel[i], s * s * bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:, k] = act[
                        kk, :, st * ii : ksz + st * ii, st * jj : ksz + st * jj
                    ].reshape(-1)
                    k += 1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k = 0
            s = compute_conv_output_size(map_list[i], 1, stride_list[i])
            mat = np.zeros((1 * 1 * in_channel[i], s * s * bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:, k] = act[
                            kk, :, st * ii : 1 + st * ii, st * jj : 1 + st * jj
                        ].reshape(-1)
                        k += 1
            mat_sc_list.append(mat)

    ik = 0
    for i in range(len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6, 10, 14]:
            mat_final.append(mat_sc_list[ik])
            ik += 1

    return mat_final, grad_list

def get_space_and_grad(
    mat_list,
    threshold,
    memory,
    task_name,
    task_name_list,
    task_id,
    space_list_all,
):
    """
    Get the space for each layer
    """
    # print("Threshold:{}".format(threshold))
    Ours = True
    if task_id == 0:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            # gradient = grad_list[i]

            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1

            # save into memory
            memory[task_name][str(i)]["space_list"] = U[:, 0:r]
            # memory[task_name][str(i)]['grad_list'] = gradient

            space_list_all.append(U[:, 0:r])

    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]

            if Ours:
                # =1. calculate the projection using previous space
                # print("activation shape:{}".format(activation.shape))
                # print("space shape:{}".format(space_list_all[i].shape))
                # delta = np.dot(np.dot(space_list_all[i],space_list_all[i].transpose()),activation)
                delta = []
                R2 = np.dot(activation, activation.transpose())
                for ki in range(space_list_all[i].shape[1]):
                    space = space_list_all[i].transpose()[ki]
                    # print(space.shape)
                    delta_i = np.dot(np.dot(space.transpose(), R2), space)
                    # print(delta_i)
                    delta.append(delta_i)
                delta = np.array(delta)

                # =2  following the GPM to get the sigma (S**2)
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()

                act_hat = activation

                act_hat -= np.dot(np.dot(space_list_all[i], space_list_all[i].transpose()), activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                sigma = S**2

                # =3 stack delta and sigma in a same list, then sort in descending order
                stack = np.hstack((delta, sigma))  # [0,..30, 31..99]
                stack_index = np.argsort(stack)[::-1]  # [99, 0, 4,7...]
                # print('stack index:{}'.format(stack_index))
                stack = np.sort(stack)[::-1]

                # =4 select the most import basis
                r_pre = len(delta)
                r = 0
                accumulated_sval = 0
                for ii in range(len(stack)):
                    if accumulated_sval < threshold[i] * sval_total:
                        accumulated_sval += stack[ii]
                        r += 1
                        if r == activation.shape[0]:
                            break
                    else:
                        break
                # if r == 0:
                #     print ('Skip Updating GPM for layer: {}'.format(i+1))
                #     continue
                # print("threshold for selecting:{}".format(np.linalg.norm(activation) ** 2))
                # print("total ranking r = {}".format(r))

                # =5 save the corresponding space
                Ui = np.hstack((space_list_all[i], U))
                sel_index = stack_index[:r]
                # print('sel_index:{}'.format(sel_index))
                # this is the current space
                U_new = Ui[:, sel_index]
                # calculate how many space from current new task
                sel_index_from_U = sel_index[sel_index > r_pre]
                # print(sel_index)
                # print(sel_index_from_U)
                if len(sel_index_from_U) > 0:
                    # update the overall space without overlap
                    total_U = np.hstack((space_list_all[i], Ui[:, sel_index_from_U]))
                    space_list_all[i] = total_U
                else:
                    space_list_all[i] = np.array(space_list_all[i])
                # else:
                #     continue
                # print("Ui shape:{}".format(Ui.shape))
                print("the number of space for current task:{}".format(r))
                print("the new increased space:{}, the threshold for new space:{}".format(len(sel_index_from_U), r_pre))

                memory[task_name][str(i)]["space_list"] = Ui[:, sel_index]


    return space_list_all


def grad_proj_cond(
    net,
    x,
    y,
    memory,
    task_name,
    task_id,
    task_name_list,
    device,
    optimizer,
    criterion,
):
    """
    get the regime descision
    """

    # calcuate the gradient for current task before training
    steps = 1
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[0:125]  # Take 125*10 random samples
    b = b.cpu()
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)

    batch_list = [2 * 12, 100, 100, 125, 125]
    grad_list = []  # list contains gradient of each layer
    act_key = list(net.act.keys())
    # print('task id:{}'.format(task_id))
    optimizer.zero_grad()
    example_out = net(example_data)

    loss = criterion(example_out[task_id], target)
    loss.backward()

    k_linear = 0
    for k, (m, params) in enumerate(net.named_parameters()):
        if "weight" in m and "bn" not in m:
            if len(params.shape) == 4:

                grad = params.grad.data.detach().cpu().numpy()
                grad = grad.reshape(grad.shape[0], grad.shape[1] * grad.shape[2] * grad.shape[3])
                grad_list.append(grad)
            else:
                if "fc3" in m and k_linear == task_id:
                    grad = params.grad.data.detach().cpu().numpy()
                    grad_list.append(grad)
                    k_linear += 1
                elif "fc3" not in m:
                    grad = params.grad.data.detach().cpu().numpy()
                    grad_list.append(grad)

    # project on each task subspace
    gradient_norm_lists_tasks = []
    for task_index in range(task_id):
        projection_norm_lists = []

        for i in range(len(grad_list)):  # layer
            space_list = memory[task_name_list[task_index]][str(i)]["space_list"]
            # print(
            #     "Task:{}, layer:{}, space shape:{}".format(
            #         task_index, i, space_list.shape
            #     )
            # )
            # grad_list is the grad for current task
            projection = np.dot(grad_list[i], np.dot(space_list, space_list.transpose()))
            projection_norm = np.linalg.norm(projection)

            projection_norm_lists.append(projection_norm)
            gradient_norm = np.linalg.norm(grad_list[i])
            # print(
            #     "Task:{}, Layer:{}, project_norm:{}, threshold for regime 1:{}".format(
            #         task_index, i, projection_norm, Eplison_1 * gradient_norm
            #     )
            # )

            # make decision if Regime 1
            # logging.info('project_norm:{}, threshold for regime 1:{}'.format(projection_norm, eplison_1 * gradient_norm))
            if projection_norm <= Eplison_1 * gradient_norm:
                memory[task_name][str(i)]["regime"][task_index] = "1"
            else:

                memory[task_name][str(i)]["regime"][task_index] = "2"
        gradient_norm_lists_tasks.append(projection_norm_lists)
        # for i in range(len(grad_list)):
        #     print(
        #         "Layer:{}, Regime:{}".format(
        #             i, memory[task_name][str(i)]["regime"][task_index]
        #         ),
        #     )

    print("-" * 20)
    print("selected top-2 tasks:")
    if task_id == 1:
        for i in range(len(grad_list)):
            memory[task_name][str(i)]["selected_task"] = [0]
    else:
        k = 2

        for layer in range(len(grad_list)):
            task_norm = []
            for t in range(len(gradient_norm_lists_tasks)):
                norm = gradient_norm_lists_tasks[t][layer]
                task_norm.append(norm)
            task_norm = np.array(task_norm)
            idx = np.argpartition(task_norm, -k)[-k:]
            memory[task_name][str(layer)]["selected_task"] = idx
            print("Layer:{}, selected task ID:{}".format(layer, memory[task_name][str(layer)]["selected_task"]))

def grad_proj_cond_Resnet18(
    net,
    x,
    y,
    memory,
    task_name,
    task_id,
    task_name_list,
    device,
    optimizer,
    criterion,
):

    # calcuate the gradient for current task before training
    steps = 1
    r = np.arange(x.size(0))

    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[0 : 100 * 5]  # Take 125*10 random samples
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)

    grad_list = []  # list contains gradient of each layer
    act_key = list(net.act.keys())
    # print('task id:{}'.format(task_id))

    optimizer.zero_grad()
    example_out = net(example_data)

    loss = criterion(example_out[task_id], target)
    loss.backward()
    # optimizer.step()
    k_conv = 0
    for k, (m, params) in enumerate(net.named_parameters()):
        if len(params.shape) == 4 and "weight" in m:

            grad = params.grad.data.detach().cpu().numpy()
            grad = grad.reshape(grad.shape[0], grad.shape[1] * grad.shape[2] * grad.shape[3])
            grad_list.append(grad)
            k_conv += 1

    # project on each task subspace
    gradient_norm_lists_tasks = []
    for task_index in range(task_id):
        projection_norm_lists = []

        for i in range(len(grad_list)):  # layer
            space_list = memory[task_name_list[task_index]][str(i)]["space_list"]
            # grad_list is the grad for current task
            projection = np.dot(
                grad_list[i], np.dot(space_list, space_list.transpose())
            )

            projection_norm = np.linalg.norm(projection)

            projection_norm_lists.append(projection_norm)
            gradient_norm = np.linalg.norm(grad_list[i])

            if projection_norm <= Eplison_1 * gradient_norm:
                memory[task_name][str(i)]["regime"][task_index] = "1"
            else:

                memory[task_name][str(i)]["regime"][task_index] = "2"

        gradient_norm_lists_tasks.append(projection_norm_lists)

    # select top-k related tasks according to the projection norm, k = 2 in general (k= 1 for task 2)
    if task_id == 1:
        for i in range(len(grad_list)):
            memory[task_name][str(i)]["selected_task"] = [0]
    else:
        if task_id == 2:
            for layer in range(len(grad_list)):
                memory[task_name][str(layer)]["selected_task"] = [1]
        else:
            k = 2

            for layer in range(len(grad_list)):
                task_norm = []
                for t in range(len(gradient_norm_lists_tasks)):
                    norm = gradient_norm_lists_tasks[t][layer]
                    task_norm.append(norm)
                task_norm = np.array(task_norm)
                idx = np.argpartition(task_norm, -k)[-k:]
                memory[task_name][str(layer)]["selected_task"] = idx