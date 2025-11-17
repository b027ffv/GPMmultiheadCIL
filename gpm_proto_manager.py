# ファイル名: GPM-based/gpm_proto_manager.py

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MultivariateNormal

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

    def compute_prototypes(self, model, data, task_id):
        """現在のタスクのプロトタイプを計算して保存する"""
        model.eval()
        
        x = data[task_id]['train']['x'].to(self.device)
        y = data[task_id]['train']['y'].to(self.device)
        
        # このタスクのグローバルクラスID (例: 10, 11, ..., 19)
        task_classes = np.unique(y.cpu().numpy()) + self.offsets[task_id]
        
        with torch.no_grad():
            # 全データの特徴量を一度に抽出
            # (メモリが足りない場合はバッチ処理に変更が必要)
            try:
                # model.forward は出力リストを返すが、特徴量は返さない。
                # GPMのモデル(model.py)を変更し、特徴量も返すようにする必要がある。
                # ここでは仮に model.features(x) で特徴量が取れると仮定する。
                # ※ GPMのAlexNet/ResNet18は self.act に活性化を保存するが、
                #   最終特徴量を取得する公開メソッドがないため、
                #   model.py の forward を修正するのが最もクリーン。
                
                # --- GPMモデル(model.py)の修正案 ---
                # AlexNet/ResNet18 の forward の最後に、
                # x (fc2またはavg_pool2dの出力) を y と一緒に返す
                # return y, x (または y, out)
                # 
                # 修正が難しい場合、暫定的に最後の層の活性化を取得する
                """output_list = model(x)
                if isinstance(model, torch.nn.DataParallel):
                    act_key = list(model.module.act.keys())[-1]
                    features = model.module.act[act_key]
                else:
                    act_key = list(model.act.keys())[-1]
                    features = model.act[act_key]"""
                output_list, features = model(x)
                    
            except Exception as e:
                print(f"特徴量の取得に失敗しました: {e}")
                print("GPM-based/model.py の forward が特徴量を返すように修正するか、")
                print("または上記 features = model.act[act_key] のロジックが正しいか確認してください。")
                return

        for global_class_id in task_classes:
            local_class_id = global_class_id - self.offsets[task_id]
            mask = (y == local_class_id)
            
            if mask.sum() == 0:
                continue
                
            feature_classwise = features[mask]
            
            # 平均と共分散を計算
            proto = feature_classwise.mean(dim=0)
            cov = torch.cov(feature_classwise.T)
            
            # ノイズを加えて正定値行列にする
            cov += 1e-5 * torch.eye(cov.shape[0]).to(self.device)
            
            # 保存
            self.prototypes[global_class_id] = proto
            self.covariances[global_class_id] = cov
            self.gaussians[global_class_id] = MultivariateNormal(proto.cpu(), cov.cpu())
            
            if global_class_id not in self.class_labels:
                self.class_labels.append(global_class_id)
                
        print(f"Task {task_id}: プロトタイプを計算・保存しました。")

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