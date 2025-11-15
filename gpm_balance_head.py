# ファイル名: GPM-based/gpm_balance_head.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from tqdm import tqdm

def balance_classifier(args, model, proto_manager, data, task_id, taskcla):
    """
    EFC++のプロトタイプ・リバランシングを実行する
    バックボーンを固定し、全ヘッドを再訓練する
    """
    print("--- プロトタイプ・リバランシング開始 ---")
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    # 1. バックボーンを固定 (GPMのモデルはbn層を持たない or track_running_stats=False が多いが念のため)
    model.eval()
    
    # 2. ヘッドのみを訓練モードにし、オプティマイザを定義
    # GPMのモデル(model.py)は fc3 (ModuleList) がヘッドに相当
    head_params = []
    if isinstance(model, torch.nn.DataParallel):
        head_module = model.module.fc3
    else:
        head_module = model.fc3
        
    for param in head_module.parameters():
        param.requires_grad = True
    head_params.extend(head_module.parameters())

    # EFC++はSGDを使用
    # argsに --lr_balance (例: 0.001) と --epochs_balance (例: 20) を追加する必要がある
    lr = args.lr_balance if hasattr(args, 'lr_balance') else 1e-3
    n_epochs = args.epochs_balance if hasattr(args, 'epochs_balance') else 20
    
    optimizer = torch.optim.SGD(head_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    
    # 3. リバランシング用データセットの準備
    
    # 3a. 過去タスクの擬似サンプル (プロトタイプから生成)
    # (ここでは全タスクのプロトタイプからサンプリングする)
    proto_samples, proto_labels = proto_manager.generate_samples_from_proto(n_samples_per_class=50)

    # 3b. 現タスクの実データ (特徴量)
    # 現タスクの訓練データから特徴量を抽出
    current_features_list = []
    current_labels_list = []
    xtrain = data[task_id]['train']['x'].to(device)
    ytrain = data[task_id]['train']['y'].to(device)
    
    with torch.no_grad():
        r = np.arange(xtrain.size(0))
        for i in range(0, len(r), args.batch_size_test):
            b = r[i : i + args.batch_size_test]
            # model.forward を修正した場合 (y, features = model(x))
            # _, features_batch = model(xtrain[b]) 
            
            # GPMの活性化 (暫定)
            _ = model(xtrain[b])
            if isinstance(model, torch.nn.DataParallel):
                act_key = list(model.module.act.keys())[-1]
                features_batch = model.module.act[act_key]
            else:
                act_key = list(model.act.keys())[-1]
                features_batch = model.act[act_key]

            current_features_list.append(features_batch)
            current_labels_list.append(ytrain[b] + proto_manager.offsets[task_id]) # グローバルラベルに

    current_features = torch.cat(current_features_list, dim=0)
    current_labels = torch.cat(current_labels_list, dim=0)

    # 3c. データセットの結合
    if proto_samples is not None:
        features_dataset = TensorDataset(current_features, current_labels)
        proto_dataset = TensorDataset(proto_samples, proto_labels)
        complete_dataset = ConcatDataset([features_dataset, proto_dataset])
    else:
        complete_dataset = TensorDataset(current_features, current_labels)
        
    loader = DataLoader(complete_dataset, batch_size=args.batch_size_train, shuffle=True)
    
    # 4. リバランシング訓練ループ
    for epoch in range(n_epochs):
        total_loss = 0
        for features_batch, labels_batch in loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # GPMモデル(model.py)のヘッド部分のみを実行
            # fc3 は ModuleList なので、各ヘッドの出力をリストで受け取る
            outputs_list = []
            for t in range(task_id + 1):
                outputs_list.append(head_module[t](features_batch))
                
            # 擬似シングルヘッド化
            outputs_global = torch.cat(outputs_list, dim=1)
            
            loss = criterion(outputs_global, labels_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"リバランシング Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(loader):.4f}")

    # 5. バックボーンの固定を解除
    for param in model.parameters():
        # fc3 以外 (バックボーン) の requires_grad を元に戻す
        # GPMは元々全パラメータが True のはず
        param.requires_grad = True
        
    model.train() # モデル全体を訓練モードに戻す
    print("--- プロトタイプ・リバランシング完了 ---")