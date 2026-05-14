import math
from math import log

class RDPAccountant:

    def __init__(self, layer_names, orders=[10], delta=1e-05):
        self.orders = orders
        self.delta = delta
        self.layer_names = layer_names
        self.dp_epsilon_total = 0.0
        self.rdp = {layer: {alpha: 0.0 for alpha in self.orders} for layer in layer_names}
        self.cumulative_rdp = {alpha: 0.0 for alpha in self.orders}
        self.global_rdp = {alpha: 0.0 for alpha in self.orders}

    def update(self, layer_name, sigma, sensitivity=1.0, count=1):
        for alpha in self.orders:
            rdp_value = self._compute_rdp_gaussian(sigma, alpha, sensitivity=sensitivity)
            self.rdp[layer_name][alpha] += count * rdp_value

    def _compute_rdp_gaussian(self, sigma, alpha, sensitivity=1.0):
        return alpha * sensitivity ** 2 / (2 * sigma ** 2)

    def get_rdp(self, layer_name):
        return self.rdp[layer_name]

    def get_epsilon(self, layer_name):
        epsilons = []
        for alpha in self.orders:
            rdp_val = self.rdp[layer_name][alpha]
            eps = rdp_val + log(1 / self.delta) / (alpha - 1)
            epsilons.append((eps, alpha))
        return min(epsilons, key=lambda x: x[0])

    def get_all_epsilons(self):
        return {layer: self.get_epsilon(layer) for layer in self.rdp}

    def accumulate_step_per_layer(self, layer_sigmas, layer_sensitivities=None, steps=1):
        if layer_sensitivities is None:
            layer_sensitivities = {layer: 1.0 for layer in self.layer_names}
        for layer_name, sigma in layer_sigmas.items():
            if layer_name not in self.layer_names:
                raise ValueError(f'Layer {layer_name} is not in the known layer list')
            sensitivity = layer_sensitivities.get(layer_name, 1.0)
            self.update(layer_name, sigma=sigma, sensitivity=sensitivity, count=steps)

    def get_total_rdp(self, delta=None):
        if delta is None:
            delta = self.delta
        total_rdp = {alpha: 0.0 for alpha in self.orders}
        for layer in self.rdp:
            for alpha in self.orders:
                total_rdp[alpha] += self.rdp[layer].get(alpha, 0.0)
        epsilons = []
        for alpha in self.orders:
            eps = total_rdp[alpha] + math.log(1 / delta) / (alpha - 1)
            epsilons.append((eps, alpha))
        eps, best_alpha = min(epsilons, key=lambda x: x[0])
        return eps

    def accumulate_and_get_rdp_increment_and_total(self, layer_sigmas, layer_sensitivities=None, steps=1, delta=None):
        if delta is None:
            delta = self.delta
        prev_total_rdp = {alpha: 0.0 for alpha in self.orders}
        for layer in self.rdp:
            for alpha in self.orders:
                prev_total_rdp[alpha] += self.rdp[layer][alpha]
        self.accumulate_step_per_layer(layer_sigmas=layer_sigmas, layer_sensitivities=layer_sensitivities, steps=steps)
        total_rdp = {alpha: 0.0 for alpha in self.orders}
        for layer in self.rdp:
            for alpha in self.orders:
                total_rdp[alpha] += self.rdp[layer][alpha]
        increment_rdp = {alpha: total_rdp[alpha] - prev_total_rdp[alpha] for alpha in self.orders}

        def calc_epsilon(rdp_dict):
            epsilons = []
            for alpha in self.orders:
                eps = rdp_dict[alpha] + math.log(1 / delta) / (alpha - 1)
                epsilons.append(eps)
            return min(epsilons)
        increment_epsilon = calc_epsilon(increment_rdp)
        total_epsilon = calc_epsilon(total_rdp)
        return (increment_rdp, increment_epsilon, total_rdp, total_epsilon)

    def accumulate_global_noise(self, sigma, sensitivity=1.0, steps=1):
        for alpha in self.orders:
            rdp_value = alpha * sensitivity ** 2 / (2 * sigma ** 2)
            self.global_rdp[alpha] += steps * rdp_value

    def accumulate_and_get_global_rdp_increment_and_total(self, sigma, sensitivity=1.0, steps=1, delta=None):
        if delta is None:
            delta = self.delta
        prev_global_rdp = self.global_rdp.copy()
        self.accumulate_global_noise(sigma=sigma, sensitivity=sensitivity, steps=steps)
        increment_rdp = {alpha: self.global_rdp[alpha] - prev_global_rdp[alpha] for alpha in self.orders}

        def calc_epsilon(rdp_dict):
            epsilons = []
            for alpha in self.orders:
                eps = rdp_dict[alpha] + math.log(1 / delta) / (alpha - 1)
                epsilons.append(eps)
            return min(epsilons)
        increment_epsilon = calc_epsilon(increment_rdp)
        total_epsilon = calc_epsilon(self.global_rdp)
        return (increment_rdp, increment_epsilon, self.global_rdp.copy(), total_epsilon)

class DPAccountant:

    def __init__(self, dp_delta=1e-05):
        self.total_epsilon = 0.0
        self.dp_delta = dp_delta

    def accumulate(self, sensitivity, sigma, steps=1):
        current_epsilon = steps * (sensitivity / sigma)
        self.total_epsilon += current_epsilon
        return (current_epsilon, self.total_epsilon)

    def exceeds_budget(self, max_epsilon):
        return self.total_epsilon > max_epsilon
