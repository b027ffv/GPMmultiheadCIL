import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal

class TaskSelector(nn.Module):
    """
    特徴量からタスクIDを予測する軽量ネットワーク
    """
    def __init__(self, input_dim, hidden_dim=512, max_tasks=20):
        super(TaskSelector, self).__init__()
        self.input_dim = input_dim
        self.max_tasks = max_tasks
        
        # シンプルな2層MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, max_tasks) # 出力は最大タスク数分確保（または可変にする）
        )

    def forward(self, x):
        return self.net(x)

def update_proto_statistics(args, model, proto_manager, data, task_id):
    """
    ProtoManagerに平均と共分散を正しく登録するためのヘルパー関数
    (gpm_proto_manager.pyのcompute_prototypesが不完全な場合に備えて)
    """
    model.eval()
    device = next(model.parameters()).device
    
    features_dict = {} # class_id -> list of features
    
    # 現在のタスクのデータを取得
    x_train = data[task_id]['train']['x'].to(device)
    y_train = data[task_id]['train']['y'].to(device)
    
    # 特徴量抽出
    with torch.no_grad():
        r = np.arange(x_train.size(0))
        for i in range(0, len(r), args.batch_size_test):
            b = r[i : i + args.batch_size_test]
            x_batch = x_train[b]
            y_batch = y_train[b]
            
            # model.pyの修正に合わせて (logits, features) を受け取る
            _, feats = model(x_batch)
            
            # 必要なら正規化（GPMの場合は生のほうが分布を捉えやすい場合もあるが、安定性のために正規化推奨）
            # feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            
            for f, y in zip(feats, y_batch):
                global_y = y.item() + proto_manager.offsets[task_id]
                if global_y not in features_dict:
                    features_dict[global_y] = []
                features_dict[global_y].append(f.cpu())

    # 平均と共分散を計算してProtoManagerに登録
    for cls, feats in features_dict.items():
        feats_tensor = torch.stack(feats).to(device) # [N, Dim]
        
        # 平均
        mean = feats_tensor.mean(dim=0)
        proto_manager.prototypes[cls] = mean
        
        # 共分散 (対角成分のみで近似するか、完全行列か。ここでは対角+微小ノイズ)
        var = feats_tensor.var(dim=0) + 1e-5
        cov = torch.diag(var)
        proto_manager.covariances[cls] = cov
        
        # サンプリング用分布作成
        try:
            dist = MultivariateNormal(mean, covariance_matrix=cov)
            proto_manager.gaussians[cls] = dist
        except Exception as e:
            print(f"Warning: Failed to create Gaussian for class {cls}. {e}")

    print(f"Task {task_id}: Prototypes and Statistics updated.")

def train_task_selector(args, model, task_selector, proto_manager, data, task_id, epochs=20):
    """
    タスク選択ネットワークを学習する
    Current Task: 実データの特徴量
    Old Tasks: ProtoManagerから生成した擬似特徴量
    """
    print(f"--- Training Task Selector for Task {task_id} ---")
    
    model.eval()
    task_selector.train()
    
    device = next(model.parameters()).device
    optimizer = optim.Adam(task_selector.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 1. 現在のタスクの特徴量を収集
    current_feats = []
    x_train = data[task_id]['train']['x'].to(device)
    with torch.no_grad():
        r = np.arange(x_train.size(0))
        for i in range(0, len(r), args.batch_size_test):
            b = r[i : i + args.batch_size_test]
            _, f = model(x_train[b])
            current_feats.append(f.cpu())
    current_feats = torch.cat(current_feats)
    current_labels = torch.full((current_feats.size(0),), task_id, dtype=torch.long)
    
    # 2. 過去のタスクの擬似特徴量を生成
    old_feats = []
    old_labels = []
    if task_id > 0:
        # 過去の全クラスからサンプリングし、対応するタスクIDをラベルとする
        # 各クラスから生成するサンプル数 = (現在のデータ数 / 過去のクラス総数) 程度でバランスを取る
        # 簡易的に固定数でも可
        n_samples_per_class = max(50, int(len(current_feats) / (proto_manager.offsets[task_id])))
        
        with torch.no_grad():
            samples, class_labels = proto_manager.generate_samples_from_proto(n_samples_per_class=n_samples_per_class)
            if samples is not None:
                # Class Label -> Task ID 変換
                task_ids_for_samples = []
                for c in class_labels:
                    # オフセットからタスクIDを逆算
                    t_found = 0
                    for t_idx in range(len(proto_manager.offsets)-1):
                        if proto_manager.offsets[t_idx] <= c < proto_manager.offsets[t_idx+1]:
                            t_found = t_idx
                            break
                    else:
                        t_found = len(proto_manager.offsets) - 1
                    task_ids_for_samples.append(t_found)
                
                old_feats = samples.cpu()
                old_labels = torch.tensor(task_ids_for_samples)

    # 3. データ結合
    if len(old_feats) > 0:
        train_feats = torch.cat([current_feats, old_feats], dim=0)
        train_targets = torch.cat([current_labels, old_labels], dim=0)
    else:
        train_feats = current_feats
        train_targets = current_labels
        
    # シャッフル
    perm = torch.randperm(train_feats.size(0))
    train_feats = train_feats[perm]
    train_targets = train_targets[perm]
    
    # 4. 学習ループ
    batch_size = 128
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for i in range(0, len(train_feats), batch_size):
            f_batch = train_feats[i:i+batch_size].to(device)
            t_batch = train_targets[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = task_selector(f_batch)
            
            # まだ学習していない未来のタスクIDへのロジットはマスクしても良いが
            # 単純にCrossEntropyで現在のタスクIDまでを教えれば下がるはず
            loss = criterion(outputs, t_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_acc += (preds == t_batch).sum().item()
            n_batches += 1
            
        if (epoch + 1) % 5 == 0:
            print(f"  Selector Epoch {epoch+1}/{epochs} | Loss: {total_loss/n_batches:.4f} | Acc: {total_acc/len(train_feats):.4f}")
            
    print("Task Selector Training Completed.")

def evaluate_with_selector(args, model, task_selector, data, current_task_id, taskcla):
    """
    Task Selectorを使ってタスクIDを予測し、そのタスクのヘッドで分類を行う
    """
    model.eval()
    task_selector.eval()
    device = next(model.parameters()).device
    
    total_correct = 0
    total_num = 0
    task_accuracies = []
    
    # Confusion Matrix for Task Selection
    task_conf_matrix = np.zeros((current_task_id + 1, current_task_id + 1))
    
    offsets = [0]
    for i in range(len(taskcla)-1):
        offsets.append(offsets[-1] + taskcla[i][1])

    with torch.no_grad():
        for t in range(current_task_id + 1):
            task_correct = 0
            task_num = 0
            
            x = data[t]['test']['x'].to(device)
            y = data[t]['test']['y'].to(device)
            y_global = y + offsets[t]
            
            r = np.arange(x.size(0))
            for i in range(0, len(r), args.batch_size_test):
                b = r[i : i + args.batch_size_test]
                data_batch = x[b]
                target_batch = y_global[b]
                
                # 1. 特徴量抽出
                logits_list, features = model(data_batch)
                
                # 2. タスクID予測
                task_logits = task_selector(features)
                
                # 現在学習済みのタスクIDまでのみ有効にする (未来のタスクを選ばないようにマスク)
                task_logits[:, current_task_id+1:] = -float('inf')
                
                pred_task_ids = task_logits.argmax(dim=1) # [Batch]
                
                # タスク選択混同行列の更新
                for p_tid in pred_task_ids:
                    if p_tid <= current_task_id:
                        task_conf_matrix[t, p_tid.item()] += 1
                
                # 3. 予測されたタスクのヘッドを使ってクラス分類
                final_preds = []
                for idx, p_tid in enumerate(pred_task_ids):
                    # バッチ内の1サンプルごとに、選ばれたヘッドの出力を見る
                    # logits_list[p_tid][idx] -> [N_Classes]
                    # ただし logits_list は [Task][Batch, Class] なので
                    # logits_list[p_tid.item()][idx] を取る
                    
                    pred_head_logits = logits_list[p_tid.item()][idx]
                    pred_local_class = pred_head_logits.argmax().item()
                    pred_global_class = pred_local_class + offsets[p_tid.item()]
                    final_preds.append(pred_global_class)
                
                final_preds = torch.tensor(final_preds).to(device)
                
                batch_correct = final_preds.eq(target_batch).sum().item()
                task_correct += batch_correct
                total_correct += batch_correct
                task_num += len(b)
                total_num += len(b)
            
            acc_t = 100.0 * task_correct / task_num if task_num > 0 else 0.0
            task_accuracies.append(round(acc_t, 2))

    avg_acc = 100.0 * total_correct / total_num if total_num > 0 else 0.0
    
    print("\n--- Task Selection Accuracy Matrix ---")
    # 行方向の和で割ってパーセント表示
    row_sums = task_conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    print((task_conf_matrix / row_sums * 100).round(1))
    print("--------------------------------------")
    
    return avg_acc, task_accuracies