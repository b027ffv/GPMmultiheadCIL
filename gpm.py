import numpy as np
import torch
import torch.nn.functional as F

from model import compute_conv_output_size

def analyze_head_outputs(args, model, device, data, current_task_id, taskcla):
    """
    教授の仮説を検証するための分析関数。
    入力タスクごとに、各ヘッドの平均logit値と、
    どのヘッド（グループ）が最大値を取ったかを計算し、行列で表示する。
    """
    model.eval()
    
    # オフセット計算（今回は使用しないが、構造把握のため）
    offsets = [0]
    for i in range(len(taskcla)-1):
        offsets.append(offsets[-1] + taskcla[i][1])

    num_tasks = current_task_id + 1
    
    # 分析結果を保存するリスト
    all_head_means_list = [] # 各ヘッドの平均logit値
    all_true_task_ids_list = [] # 入力画像の真のタスクID

    print(f"\n--- [タスク {current_task_id}] ヘッド出力分析（教授の仮説検証）---")

    with torch.no_grad():
        # 1. 全テストデータをバッチ処理し、ヘッドごとの平均logitを収集
        for t in range(num_tasks): # t = 真のタスクID
            # タスク t のテストデータを取得
            x = data[t]['test']['x'].to(device)
            y = data[t]['test']['y'].to(device)
            
            r = np.arange(x.size(0))
            for i in range(0, len(r), args.batch_size_test):
                b = r[i : i + args.batch_size_test]
                data_batch = x[b]
                
                # (y, features) を返すように修正済みの model.py を前提とする
                output_list, _ = model(data_batch)
                
                # 現在のタスクまでのヘッドのみを対象
                output_list_current = output_list[:num_tasks]
                
                # --- ここが仮説の核心 ---
                # 各ヘッドの出力(例:[B, 10])の「平均値」を計算
                # (B = バッチサイズ)
                head_means = [torch.mean(head_output, dim=1) for head_output in output_list_current]
                
                # [B, num_tasks] のテンソルにまとめる
                head_means_tensor = torch.stack(head_means, dim=1)
                
                all_head_means_list.append(head_means_tensor.cpu())
                
                # このバッチの真のタスクIDを保存
                true_task_ids_batch = torch.full((len(data_batch),), fill_value=t, dtype=torch.long)
                all_true_task_ids_list.append(true_task_ids_batch)

    # 2. 収集したデータを集計
    all_head_means = torch.cat(all_head_means_list, dim=0)
    all_true_task_ids = torch.cat(all_true_task_ids_list, dim=0)
    
    # 3. 分析行列を作成
    
    # 行列1: 各ヘッドの平均logit値（バイアスの可視化）
    # (行 = 入力画像の真のタスク, 列 = 出力ヘッド)
    mean_value_matrix = torch.zeros(num_tasks, num_tasks)
    
    # 行列2: グループルーティングの混乱行列
    # (行 = 入力画像の真のタスク, 列 = argmaxで選ばれたヘッド)
    routing_confusion_matrix = torch.zeros(num_tasks, num_tasks)
    
    # どのヘッド（グループ）が勝ったか
    pred_head_group = all_head_means.argmax(dim=1) # [全サンプル数]

    for true_task in range(num_tasks):
        # 真のタスクが t であるサンプルのインデックス
        mask = (all_true_task_ids == true_task)
        if mask.sum() == 0:
            continue
            
        # 1. 真のタスク t の入力に対する、全ヘッドの平均logit値
        mean_values_for_task_t = all_head_means[mask].mean(dim=0)
        mean_value_matrix[true_task, :] = mean_values_for_task_t
        
        # 2. 真のタスク t の入力が、どのヘッドグループにルーティングされたか
        preds_for_task_t = pred_head_group[mask]
        for pred_task in range(num_tasks):
            count = (preds_for_task_t == pred_task).sum().item()
            routing_confusion_matrix[true_task, pred_task] = (count / mask.sum().item()) * 100.0

    # 4. 結果の出力
    print("分析1: 平均logit値の行列 (行=入力タスク / 列=出力ヘッド)")
    print("       (対角成分が他よりも高ければ、バイアスが制御できていることを示唆)")
    print(mean_value_matrix.numpy().round(3))
    
    print("\n分析2: グループルーティング混乱行列 (行=入力タスク / 列=予測ヘッド) [%]")
    print("       (対角成分が高ければ、教授の仮説が正しいことを示す)")
    print(routing_confusion_matrix.numpy().round(2))
    print("--- 分析終了 ---")

def test_class_incremental_nme(args, model, device, data, current_task_id, taskcla, proto_manager):
    """
    クラス増分学習のためのNME（最近傍平均）評価関数
    (分類ヘッド fc3 を使わず、プロトタイプとの距離で分類する)
    """
    model.eval()
    total_correct = 0
    total_num = 0

    # 1. 全既知クラスのプロトタイプを取得
    # proto_manager.prototypes は { global_class_id: テンソル } の辞書
    if not proto_manager.prototypes:
        print("エラー: プロトタイプが計算されていません。")
        return 0.0

    # 辞書のキー（グローバルクラスID）と値（プロトタイプ）を抽出し、順序を固定
    # .items() は Python 3.7+ で挿入順を保証
    known_class_ids = list(proto_manager.prototypes.keys())
    all_prototypes = torch.stack(list(proto_manager.prototypes.values())).to(device)
    # all_prototypes の形状: [全クラス数, 特徴量次元数]

    with torch.no_grad():
        for t in range(current_task_id + 1):
            # タスク t のテストデータを取得
            x = data[t]['test']['x'].to(device)
            y = data[t]['test']['y'].to(device)
            
            # グローバルラベルに変換
            y_global = y + proto_manager.offsets[t]
            
            # バッチごとに処理
            r = np.arange(x.size(0))
            for i in range(0, len(r), args.batch_size_test):
                b = r[i : i + args.batch_size_test]
                data_batch = x[b]
                target_batch = y_global[b]
                
                # 2. モデルで特徴量を抽出 (分類ヘッドは使わない)
                # (model.py の forward が (output_list, features) を返すよう要修正)
                
                # --- GPMモデル(model.py)の修正が前提 ---
                # AlexNet/ResNet18 の forward の最後を以下のように変更
                # return y, x  (AlexNet)
                # return y, out (ResNet18)
                
                try:
                    _, features_batch = model(data_batch)
                except ValueError:
                    # GPMモデル(model.py)の修正がされていない場合のフォールバック
                    _ = model(data_batch)
                    if isinstance(model, torch.nn.DataParallel):
                        act_key = list(model.module.act.keys())[-1]
                        features_batch = model.module.act[act_key]
                    else:
                        act_key = list(model.act.keys())[-1]
                        features_batch = model.act[act_key]

                # 3. 特徴量と全プロトタイプとの距離を計算
                # (バッチサイズ, 1, 特徴量次元) と (1, 全クラス数, 特徴量次元) でユークリッド距離を計算
                dist = torch.cdist(features_batch.unsqueeze(1), all_prototypes.unsqueeze(0))
                # dist の形状: (バッチサイズ, 1, 全クラス数)
                
                dist = dist.squeeze(1) # 形状: (バッチサイズ, 全クラス数)
                
                # 4. 最も距離が近いプロトタイプの「インデックス」を取得
                pred_indices = torch.argmin(dist, dim=1)
                
                # 5. インデックスをグローバルクラスIDに変換
                pred_global_ids = torch.tensor([known_class_ids[idx] for idx in pred_indices]).to(device)

                total_correct += pred_global_ids.eq(target_batch).sum().item()
                total_num += len(b)

    acc = 100.0 * total_correct / total_num
    return acc


def test_class_incremental(args, model, device, data, current_task_id, taskcla):
    """
    EFCのようにヘッドを結合してクラス増分学習として評価する関数
    """
    model.eval()
    total_correct = 0
    total_num = 0
    task_accuracies = []
    
    # クラス数のオフセット計算用リスト (例: [0, 10, 20, ...])
    offsets = [0]
    for i in range(len(taskcla)-1):
        offsets.append(offsets[-1] + taskcla[i][1])

    # これまでに学習した全タスクのテストデータで評価
    with torch.no_grad():
        for t in range(current_task_id + 1):

            task_correct = 0
            task_num = 0
            # タスク t のテストデータを取得
            x = data[t]['test']['x'].to(device)
            y = data[t]['test']['y'].to(device)
            
            # データローダーのラベル(0-9)をグローバルラベル(例: 10-19)に変換
            y_global = y + offsets[t]
            
            # バッチごとに処理
            r = np.arange(x.size(0))
            for i in range(0, len(r), args.batch_size_test):
                b = r[i : i + args.batch_size_test]
                data_batch = x[b]
                target_batch = y_global[b]
                
                # モデルの出力を取得（全タスクのリスト）
                output_list, _ = model(data_batch)
                """
                # 現在のタスクまでのヘッドのみを結合 (Pseudo-Single Head化)
                # output_list[:current_task_id+1] を結合 dim=1
                output_global = torch.cat(output_list[:current_task_id+1], dim=1)"""

                # 現在のタスクまでのヘッドのみを対象
                output_list_current = output_list[:current_task_id+1]
                
                # --- ▼▼▼ バイアス補正の工夫（L2ノルム正規化） ▼▼▼ ---
                
                normalized_outputs = []
                for head_output in output_list_current:
                    # head_output の形状: (バッチサイズ, タスク内クラス数)
                    
                    # 1. 各サンプル(バッチ内)のlogitベクトルのL2ノルム（長さ）を計算
                    #    (dim=1 はクラス方向のノルム)
                    #    (1e-6 はゼロ除算防止のための微小値)
                    norm = torch.norm(head_output, p=2, dim=1, keepdim=True) + 1e-6
                    
                    # 2. logitベクトルをその長さで割り、ノルムを1に正規化
                    normalized_outputs.append(head_output / norm)
                
                # 3. 正規化されたlogitを結合
                output_global = torch.cat(normalized_outputs, dim=1)

                # --- ▲▲▲ 工夫ここまで ▲▲▲ ---
                
                # 全クラスの中で最大値を持つインデックスを取得
                pred = output_global.argmax(dim=1, keepdim=True)
                
                # 正解数を計算
                correct_mask = pred.eq(target_batch.view_as(pred))
                
                # ★修正: タスクごとと全体の両方で集計
                task_correct += correct_mask.sum().item()
                total_correct += correct_mask.sum().item()
                task_num += len(b)
                total_num += len(b)
            # ★追加: タスクtのバッチ処理が終わったら、タスクtの精度を計算
            if task_num > 0:
                task_acc = 100.0 * task_correct / task_num
                task_accuracies.append(round(task_acc, 2)) # 小数点以下2桁でリストに追加
            else:
                task_accuracies.append(0.0) # データがなかった場合

    # 全体の平均精度を計算
    if total_num > 0:
        acc = 100.0 * total_correct / total_num
    else:
        acc = 0.0
    return acc, task_accuracies

def train(args, model, device, x, y, optimizer, criterion, task_id):
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
        output_list, _ = model(data)
        loss = criterion(output_list[task_id], target)
        loss.backward()
        optimizer.step()


def train_projected(
    args,
    model,
    device,
    x,
    y,
    optimizer,
    criterion,
    task_id,
    feature_mat,
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
        output,_ = model(data)
        loss = criterion(output[task_id], target)
        loss.backward()
        # Gradient Projections
        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if k < 15 and len(params.size()) != 1:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), feature_mat[kk]).view(
                    params.size()
                )
                kk += 1
            elif (k < 15 and len(params.size()) == 1) and task_id != 0:
                params.grad.data.fill_(0)

        optimizer.step()


def train_projected_Resnet18(args, model, device, x, y, optimizer, criterion, task_id, feature_mat):
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
        output = model(data)
        loss = criterion(output[task_id], target)
        loss.backward()
        # Gradient Projections
        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if len(params.size()) == 4:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), feature_mat[kk]).view(
                    params.size()
                )
                kk += 1
            elif len(params.size()) == 1 and task_id != 0:
                params.grad.data.fill_(0)

        optimizer.step()


def test(args, model, device, x, y, criterion, task_id, **kwargs):
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
            output_list, _ = model(data)
            loss = criterion(output_list[task_id], target)
            pred = output_list[task_id].argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item() * len(b)
            total_num += len(b)

    acc = 100.0 * correct / total_num
    final_loss = total_loss / total_num

    return final_loss, acc


def get_representation_matrix_alexnet(net, device, x, y=None):
    # Collect activations by forward pass
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[0:125]  # Take 125 random samples
    example_data = x[b]
    example_data = example_data.to(device)
    example_out = net(example_data)

    batch_list = [2 * 12, 100, 100, 125, 125]
    mat_list = []
    act_key = list(net.act.keys())
    for i in range(len(net.map)):
        bsz = batch_list[i]
        k = 0
        if i < 3:
            ksz = net.ksize[i]
            s = compute_conv_output_size(net.map[i], net.ksize[i])
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

    # print("-" * 30)
    # print("Representation Matrix")
    # print("-" * 30)
    # for i in range(len(mat_list)):
    #     print("Layer {} : {}".format(i + 1, mat_list[i].shape))
    # print("-" * 30)
    return mat_list


def get_representation_matrix_ResNet18(net, device, x, y=None):
    # Collect activations by forward pass
    net.eval()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[0:100]  # ns=100 examples
    example_data = x[b]
    example_data = example_data.to(device)
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

    batch_list = [
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        50,
        50,
        50,
        100,
        100,
        100,
        100,
        100,
        100,
    ]  # scaled
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
                    mat[:, k] = act[kk, :, st * ii : ksz + st * ii, st * jj : ksz + st * jj].reshape(-1)
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
                        mat[:, k] = act[kk, :, st * ii : 1 + st * ii, st * jj : 1 + st * jj].reshape(-1)
                        k += 1
            mat_sc_list.append(mat)

    ik = 0
    for i in range(len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6, 10, 14]:
            mat_final.append(mat_sc_list[ik])
            ik += 1

    # print("-" * 30)
    # print("Representation Matrix")
    # print("-" * 30)
    # for i in range(len(mat_final)):
    #     print("Layer {} : {}".format(i + 1, mat_final[i].shape))
    # print("-" * 30)
    return mat_final


def update_GPM(
    model,
    mat_list,
    threshold,
    feature_list=[],
):
    # print("Threshold: ", threshold)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            feature_list.append(U[:, 0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print("Skip Updating GPM for layer: {}".format(i + 1))
                continue
            # update GPM
            Ui = np.hstack((feature_list[i], U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                feature_list[i] = Ui[:, 0 : Ui.shape[0]]
            else:
                feature_list[i] = Ui

    # print("-" * 40)
    # print("Gradient Constraints Summary")
    # print("-" * 40)
    # for i in range(len(feature_list)):
    #     print(
    #         "Layer {} : {}/{}".format(
    #             i + 1, feature_list[i].shape[1], feature_list[i].shape[0]
    #         )
    #     )
    # print("-" * 40)
    return feature_list
