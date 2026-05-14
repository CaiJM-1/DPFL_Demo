import os
import math
import numpy as np
import matplotlib.pyplot as plt
from config import Config

class GradientMonitor:

    def __init__(self):
        self.enabled = getattr(Config, 'enable_gradient_monitor', False)
        self.out_dir = os.path.join(Config.output_root, getattr(Config, 'gradient_monitor_output_dir', 'gradient_distributions'))
        self.max_components = getattr(Config, 'gradient_monitor_max_components', 5)
        self.bins = getattr(Config, 'gradient_monitor_bins', 60)
        self.freq_round = getattr(Config, 'gradient_monitor_freq_round', 1)
        self.record_epoch_level = getattr(Config, 'gradient_monitor_record_epoch_level', True)
        self.min_values = getattr(Config, 'gradient_monitor_min_values', 20)
        self.verbose = getattr(Config, 'gradient_monitor_verbose', False)
        self.first_batch_only = getattr(Config, 'gradient_monitor_first_batch_only', False)
        self.epoch_aggregate = getattr(Config, 'gradient_monitor_epoch_aggregate', False)
        self.epoch_agg_sample_max = getattr(Config, 'gradient_monitor_epoch_agg_sample_max', 50000)
        self._epoch_buffers = {}
        os.makedirs(self.out_dir, exist_ok=True)

    def record(self, model, round_idx: int, epoch_idx: int, client_id: int):
        if not self.enabled:
            return
        if self.freq_round > 1 and round_idx % self.freq_round != 0:
            return
        for name, param in model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            grad_values = param.grad.detach().cpu().view(-1).numpy()
            if grad_values.size < self.min_values:
                continue
            layer_dir = os.path.join(self.out_dir, f'client_{client_id}', name)
            if self.record_epoch_level:
                filename = f'round_{round_idx}_epoch_{epoch_idx}.png'
            else:
                filename = f'round_{round_idx}.png'
            fig_path = os.path.join(layer_dir, filename)
            if self.first_batch_only and os.path.exists(fig_path):
                if self.epoch_aggregate:
                    key = (client_id, round_idx, epoch_idx, name)
                    self._append_to_buffer(key, grad_values)
                continue
            self._process_layer(name, grad_values, round_idx, epoch_idx, client_id)
            if self.epoch_aggregate:
                key = (client_id, round_idx, epoch_idx, name)
                self._append_to_buffer(key, grad_values)

    def _process_layer(self, layer_name, grad_values, round_idx, epoch_idx, client_id):
        data = grad_values.astype(np.float64)
        mean_all = float(data.mean())
        best = self._select_best_gmm(data)
        comp_k = best['k']
        weights = best['weights']
        means = best['means']
        variances = best['variances']
        layer_dir = os.path.join(self.out_dir, f'client_{client_id}', layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        if self.record_epoch_level:
            filename = f'round_{round_idx}_epoch_{epoch_idx}.png'
        else:
            filename = f'round_{round_idx}.png'
        fig_path = os.path.join(layer_dir, filename)
        if self.first_batch_only and os.path.exists(fig_path):
            return
        self._plot_distribution(data, weights, means, variances, layer_name, round_idx, epoch_idx, fig_path)
        stats_path = os.path.join(layer_dir, 'stats.csv')
        header_needed = not os.path.exists(stats_path)
        with open(stats_path, 'a', encoding='utf-8') as f:
            if header_needed:
                f.write('round,epoch,count,mean,K,component_means,component_variances\n')
            f.write(f'{round_idx},{epoch_idx},{data.size},{mean_all:.8f},{comp_k},' + '|'.join((f'{m:.8f}' for m in means)) + ',' + '|'.join((f'{v:.8f}' for v in variances)) + '\n')
        if self.verbose:
            pass

    def _plot_distribution(self, data, weights, means, variances, layer_name, round_idx, epoch_idx, save_path):
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=self.bins, density=True, alpha=0.55, color='#4E79A7', edgecolor='none')
        x_min = float(data.min())
        x_max = float(data.max())
        span = x_max - x_min
        if span <= 0:
            span = 1.0
        x = np.linspace(x_min - 0.05 * span, x_max + 0.05 * span, 400)
        mix_pdf = np.zeros_like(x)
        for w, m, v in zip(weights, means, variances):
            std = math.sqrt(max(v, 1e-12))
            comp_pdf = 1.0 / (std * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - m) / std) ** 2)
            mix_pdf += w * comp_pdf
        plt.xlabel('Gradient Value')
        plt.ylabel('Density')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()

    def _select_best_gmm(self, data: np.ndarray):
        best = None
        for k in range(1, self.max_components + 1):
            model = self._fit_gmm_em_1d(data, k)
            if best is None or model['bic'] < best['bic']:
                best = model
        return best

    def _fit_gmm_em_1d(self, data: np.ndarray, k: int):
        n = data.size
        weights = np.full(k, 1.0 / k)
        quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
        means = np.quantile(data, quantiles)
        var_global = float(np.var(data) + 1e-08)
        variances = np.full(k, var_global)
        max_iter = 20
        tol = 1e-05
        log_likelihood_old = None
        data_matrix = data.reshape(-1, 1)
        for _ in range(max_iter):
            resp = np.zeros((n, k))
            for j in range(k):
                std = math.sqrt(max(variances[j], 1e-12))
                coef = 1.0 / (std * math.sqrt(2 * math.pi))
                resp[:, j] = weights[j] * coef * np.exp(-0.5 * ((data - means[j]) / std) ** 2)
            total = resp.sum(axis=1, keepdims=True)
            total[total <= 0] = 1e-12
            resp /= total
            Nk = resp.sum(axis=0)
            weights = Nk / n
            means = (resp * data_matrix).sum(axis=0) / Nk
            for j in range(k):
                diff = data - means[j]
                variances[j] = (resp[:, j] * diff * diff).sum() / Nk[j] + 1e-12
            ll = 0.0
            for i in range(n):
                s = 0.0
                for j in range(k):
                    std = math.sqrt(max(variances[j], 1e-12))
                    coef = 1.0 / (std * math.sqrt(2 * math.pi))
                    s += weights[j] * coef * math.exp(-0.5 * ((data[i] - means[j]) / std) ** 2)
                ll += math.log(max(s, 1e-12))
            if log_likelihood_old is not None and abs(ll - log_likelihood_old) < tol:
                break
            log_likelihood_old = ll
        p = k - 1 + k + k
        bic = -2 * log_likelihood_old + p * math.log(n)
        return {'k': k, 'weights': weights, 'means': means, 'variances': variances, 'bic': bic, 'log_likelihood': log_likelihood_old}

    def begin_epoch(self, round_idx: int, epoch_idx: int, client_id: int):
        if not self.enabled or not self.epoch_aggregate:
            return
        keys = [k for k in self._epoch_buffers.keys() if k[0] == client_id and k[1] == round_idx and (k[2] == epoch_idx)]
        for k in keys:
            self._epoch_buffers.pop(k, None)

    def accumulate(self, model, round_idx: int, epoch_idx: int, client_id: int):
        if not self.enabled or not self.epoch_aggregate:
            return
        for name, param in model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            values = param.grad.detach().cpu().view(-1).numpy().astype(np.float64)
            if values.size < self.min_values:
                continue
            key = (client_id, round_idx, epoch_idx, name)
            self._append_to_buffer(key, values)

    def _append_to_buffer(self, key, values: np.ndarray):
        if key not in self._epoch_buffers:
            self._epoch_buffers[key] = {'chunks': [], 'size': 0}
        entry = self._epoch_buffers[key]
        entry['chunks'].append(values)
        entry['size'] += values.size
        if entry['size'] > self.epoch_agg_sample_max:
            merged = np.concatenate(entry['chunks'])
            idx = np.random.choice(merged.size, self.epoch_agg_sample_max, replace=False)
            sampled = merged[idx]
            entry['chunks'] = [sampled]
            entry['size'] = sampled.size

    def flush_epoch(self, round_idx: int, epoch_idx: int, client_id: int):
        if not self.enabled or not self.epoch_aggregate:
            return
        layer_keys = [k for k in self._epoch_buffers.keys() if k[0] == client_id and k[1] == round_idx and (k[2] == epoch_idx)]
        for key in layer_keys:
            _, _, _, layer_name = key
            entry = self._epoch_buffers.pop(key)
            if entry is None:
                continue
            if isinstance(entry, dict) and 'chunks' in entry:
                if len(entry['chunks']) == 0:
                    continue
                data = np.concatenate(entry['chunks'])
            else:
                data = entry
            if data is None or data.size < self.min_values:
                continue
            best = self._select_best_gmm(data)
            weights, means, variances = (best['weights'], best['means'], best['variances'])
            layer_dir = os.path.join(self.out_dir, f'client_{client_id}', layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            fig_path = os.path.join(layer_dir, f'round_{round_idx}_epoch_{epoch_idx}_agg.png')
            self._plot_distribution(data, weights, means, variances, layer_name, round_idx, epoch_idx, fig_path)
            stats_path = os.path.join(layer_dir, 'stats_agg.csv')
            header_needed = not os.path.exists(stats_path)
            with open(stats_path, 'a', encoding='utf-8') as f:
                if header_needed:
                    f.write('round,epoch,count,mean,K,component_means,component_variances\n')
                f.write(f'{round_idx},{epoch_idx},{data.size},{float(data.mean()):.8f},{best['k']},' + '|'.join((f'{m:.8f}' for m in means)) + ',' + '|'.join((f'{v:.8f}' for v in variances)) + '\n')
gradient_monitor = GradientMonitor()
__all__ = ['gradient_monitor']
