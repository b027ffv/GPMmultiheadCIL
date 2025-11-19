# ファイル名: GPM-based/gpm_proto_manager.py

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MultivariateNormal
import torch.nn.functional as F # 追加

class ProtoManager:
    def __init__(self, device, taskcla, batch_size_test):
        self.device = device
        self.taskcla = taskcla
        self.batch_size_test = batch_size_test
        
        # グローバルクラスIDをキーとする辞書
        self.prototypes = {}  # 平均 (mean)
        self.covariances = {} # 共分散 (covariance)
        self.gaussians = {}   # サンプリング用ガウス分布
        self.class_labels = [] # これまで見た全クラスのラベル
        
        # タスクごとのクラスオフセット (例: [0, 10, 20, ...])
        self.offsets = [0]
        for i in range(len(taskcla)-1):
            self.offsets.append(self.offsets[-1] + taskcla[i][1])
        self.covariances = {}
        self.task_energy_means = {}
        self.task_energy_stats = {} # (mean, std) を保存する辞書に変更

    def compute_prototypes(self, model, data, task_id):
        """プロトタイプと平均エネルギーを計算"""
        model.eval()
        
        x_all = data[task_id]['train']['x']
        y_all = data[task_id]['train']['y']
        
        features_list = []
        energy_list = [] # ★追加
        
        batch_size = self.batch_size_test
        r = np.arange(x_all.size(0))
        
        with torch.no_grad():
            for i in range(0, len(r), batch_size):
                b = r[i : i + batch_size]
                x_batch = x_all[b].to(self.device)
                
                # model.py修正済み: (y, features) が返る
                # output_list: [Logits_Task0, Logits_Task1, ...]
                output_list, features_batch = model(x_batch)
                
                # 特徴量保存 (NME用)
                # features_batch = features_batch.clamp(min=1e-6).sqrt() # 必要ならPowerNorm
                features_batch = F.normalize(features_batch, p=2, dim=1)
                features_list.append(features_batch.cpu())
                
                # ★追加: 現在のタスク(task_id)のヘッドのエネルギーを計算して保存
                logits = output_list[task_id]
                lse = torch.logsumexp(logits, dim=1) # [Batch]
                energy_list.append(lse.cpu())

        # --- プロトタイプ計算 (既存処理) ---
        features_all = torch.cat(features_list, dim=0).to(self.device)
        y_all = y_all.to(self.device)
        # ... (既存のプロトタイプ計算ロジック: 平均とって保存) ...
        # ...
        
        # ★追加: 平均エネルギーの計算と保存
        all_energies = torch.cat(energy_list, dim=0)
        mean_energy = all_energies.mean().item()
        std_energy = all_energies.std().item()

        # 標準偏差が0だと割れないので、念のため最小値を設定
        if std_energy < 1e-6:
            std_energy = 1.0
        self.task_energy_stats[task_id] = (mean_energy, std_energy)
        
        print(f"Task {task_id}: Energy Stats Updated. Mean={mean_energy:.4f}, Std={std_energy:.4f}")

    def generate_samples_from_proto(self, n_samples_per_class=50):
        """全過去クラスからプロトタイプを使って擬似サンプルを生成"""
        all_samples = []
        all_labels = []
        
        if not self.gaussians:
            return None, None

        for global_class_id, dist in self.gaussians.items():
            samples = dist.sample((n_samples_per_class,)).to(self.device)
            labels = torch.full((n_samples_per_class,), global_class_id, dtype=torch.long).to(self.device)
            
            all_samples.append(samples)
            all_labels.append(labels)
            
        return torch.cat(all_samples, dim=0), torch.cat(all_labels, dim=0)