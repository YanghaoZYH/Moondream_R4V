import time
import numpy as np
import torch

from scipy.stats import chi2, beta, norm, rankdata

try:
    from numba import njit
except ImportError:
    njit = None


def _compute_ess_per_chain_numpy(chains: np.ndarray, max_lag: int):
    T, M = chains.shape
    ess = np.empty(M, dtype=np.float64)

    for m in range(M):
        x = chains[:, m].astype(np.float64)
        x = x - np.mean(x)

        std = np.std(x, ddof=0)
        if std < 1e-14:
            ess[m] = 1.0
            continue

        x = x / std
        var = np.var(x, ddof=0)

        rho_sum = 0.0
        for lag in range(1, max_lag + 1):
            v1 = x[:-lag]
            v2 = x[lag:]
            acov = np.mean(v1 * v2)
            rho = acov / var

            if not np.isfinite(rho) or rho <= 0:
                break
            rho_sum += rho

        tau_int = 1.0 + 2.0 * rho_sum
        ess[m] = max(1.0, T / tau_int)

    return ess


if njit is not None:
    @njit(cache=True)
    def _compute_ess_per_chain_numba(chains, max_lag):
        T, M = chains.shape
        ess = np.empty(M, dtype=np.float64)

        for m in range(M):
            x = chains[:, m].astype(np.float64).copy()

            mean_x = 0.0
            for t in range(T):
                mean_x += x[t]
            mean_x /= T
            for t in range(T):
                x[t] -= mean_x

            var0 = 0.0
            for t in range(T):
                var0 += x[t] * x[t]
            var0 /= T
            std = np.sqrt(var0)

            if std < 1e-14:
                ess[m] = 1.0
                continue

            for t in range(T):
                x[t] /= std

            var = 0.0
            for t in range(T):
                var += x[t] * x[t]
            var /= T

            rho_sum = 0.0
            for lag in range(1, max_lag + 1):
                acov = 0.0
                count = T - lag
                for t in range(count):
                    acov += x[t] * x[t + lag]
                acov /= count
                rho = acov / var

                if not np.isfinite(rho) or rho <= 0.0:
                    break
                rho_sum += rho

            tau_int = 1.0 + 2.0 * rho_sum
            ess_val = T / tau_int
            if ess_val < 1.0:
                ess_val = 1.0
            ess[m] = ess_val

        return ess


def repp_upper_bound(
    Y_all_flatten: np.ndarray,
    chain_indices_flatten: np.ndarray,
    m_total: int,
    N: int,
    alpha: float = 1e-15,
):
    """
    High-probability upper bound for REPP.
    Here N must be the number of independent certification chains.
    """
    assert m_total == len(Y_all_flatten)
    assert len(chain_indices_flatten) == len(Y_all_flatten)

    if m_total == 0:
        return {
            "p_hat_point": 1.0,
            "log_p_point": 0.0,
            "U": 1.0,
            "logU": 0.0,
            "P_hat": 0,
            "theta_L": 0.0,
            "J_eff": 0,
            "max_w_eff": 0,
            "sum_log_DeltaU": 0.0,
        }

    # 1) Point estimator (MVUE) in output space
    _, raw_counts = np.unique(Y_all_flatten, return_counts=True)
    log_p_point = np.sum(np.log((N - 1) / (N - 1 + raw_counts.astype(np.float64))))
    p_hat_point = np.exp(log_p_point)

    # 2) High-probability upper bound

    # A. Geometric failures
    unique_vals, inv = np.unique(Y_all_flatten, return_inverse=True)
    r = np.bincount(inv)
    inv_and_chain = np.column_stack((inv, chain_indices_flatten))
    unique_inv_and_chain = np.unique(inv_and_chain, axis=0)
    n_d = np.bincount(
        unique_inv_and_chain[:, 0].astype(np.int64, copy=False),
        minlength=r.size,
    )
    w = r - n_d

    Dy_mask = w > 0
    w_eff = w[Dy_mask].astype(np.float64)
    J_eff = int(w_eff.size)

    # B. Poisson component count
    P_hat = int(m_total) - int(np.sum(w_eff))
    if P_hat < 0:
        P_hat = 0

    # C. Split alpha
    if J_eff > 0:
        alpha0 = alpha / 2.0
        alpha_d = (alpha / 2.0) / J_eff
    else:
        alpha0 = alpha
        alpha_d = 0.0

    # D. Poisson lower bound on total rate
    theta_L = 0.0
    if P_hat > 0:
        theta_L = max(0.0, 0.5 * chi2.ppf(alpha0, df=2 * P_hat))

    # E. Atomic part
    n_visiting = n_d[Dy_mask].astype(np.float64)

    log_prod_DeltaU = 0.0
    if J_eff > 0:
        DeltaU = beta.ppf(1.0 - alpha_d, a=n_visiting, b=w_eff + 1)
        DeltaU = np.clip(DeltaU, 1e-300, 1.0)
        log_prod_DeltaU = float(np.sum(np.log(DeltaU)))

    # F. Final
    logU = (-theta_L / N) + log_prod_DeltaU
    U = np.exp(logU)

    out = {
        "p_hat_point": p_hat_point,
        "log_p_point": log_p_point,
        "U": U,
        "logU": logU,
        "P_hat": P_hat,
        "theta_L": theta_L,
        "J_eff": J_eff,
        "max_w_eff": np.max(w_eff) if len(w_eff) > 0 else 0,
        "sum_log_DeltaU": log_prod_DeltaU,
    }

    return out


class REPP_MVUE:
    def __init__(
        self,
        problem,
        nb_var,
        bounds,
        count_particles=100,
        count_mh_steps=1000,
        timeout_threshold=1800,
        tau=1e-50,
        grid_sizes=10001,
        threshold=0,
        cert_fraction=0.2,
        verify=True,
        de_ratio=0.8,
        init_width_cert=1.0,
        init_width_search=1.0,
        prior_probs=None,
        **kwargs,
    ):
        self.problem = problem
        self.verify = verify
        self.nb_var = nb_var
        self.bounds = bounds
        self.count_particles = count_particles
        self.count_mh_steps = count_mh_steps
        self.threshold = threshold
        self.timeout_threshold = timeout_threshold
        self.start_time = time.time()
        self.tau = tau

        self.cert_fraction = cert_fraction
        self.de_ratio = de_ratio
        self.init_width_cert = init_width_cert
        self.init_width_search = init_width_search

        self.grid_sizes = grid_sizes
        self.prior_probs = prior_probs

        self.snap_cert_pool = bool(kwargs.get("snap_cert_pool", True))
        self.snap_search_pool = bool(kwargs.get("snap_search_pool", True))
        self.cert_use_cm = bool(kwargs.get("cert_use_cm", True))
        self.cert_diag_burn_in = int(kwargs.get("cert_diag_burn_in", 20))
        self.cert_diag_stride = int(kwargs.get("cert_diag_stride", 10))
        self.cert_diag_max_window = int(kwargs.get("cert_diag_max_window", 200))
        self.cert_no_growth_patience = int(kwargs.get("cert_no_growth_patience", 10))
        self.use_search_sa = bool(kwargs.get("use_search_sa", True))
        self.use_eval_cache = bool(kwargs.get("use_eval_cache", False))
        self.search_sa_t0 = float(kwargs.get("search_sa_t0", 5e-2))
        self.search_sa_tmin = float(kwargs.get("search_sa_tmin", 1e-4))
        self.search_sa_decay = float(kwargs.get("search_sa_decay", 0.995))
        self.ess = float(kwargs.get("ess", 20))


        self.best_y = -np.inf
        self.best_x = None
        self.repp = np.inf
        self.query = 0
        self.log_p_point_star_true = 0.0

        self.lb, self.ub = self._set_var_bound(bounds, self.nb_var)

    @staticmethod
    def _set_var_bound(bounds, nb_var):
        bounds = np.array(bounds, dtype=float)
        if bounds.shape != (nb_var, 2):
            raise AssertionError(
                f"The shape of bounds should be ({nb_var}, 2), but got {bounds.shape}"
            )
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        if np.any(lb - ub > 0):
            raise AssertionError("Lower bound is larger than upper bound.")
        return lb, ub

    @staticmethod
    def _normalize_grid_sizes(grid_sizes, nb_var):
        if np.isscalar(grid_sizes):
            g = np.full(nb_var, int(grid_sizes), dtype=np.int64)
        else:
            g = np.asarray(grid_sizes, dtype=np.int64)
            if g.shape != (nb_var,):
                raise AssertionError(
                    f"grid_sizes should be a scalar or shape ({nb_var},), but got {g.shape}"
                )
        if np.any(g < 2):
            raise AssertionError("Each grid size must be >= 2.")
        return g

    def _build_grid_info(self, device, dtype):
        self.grid_sizes_np = self._normalize_grid_sizes(self.grid_sizes, self.nb_var)
        self.grid_sizes_t = torch.tensor(self.grid_sizes_np, device=device, dtype=torch.long)
        self.delta_t = (self.ub_t - self.lb_t) / (self.grid_sizes_t.to(dtype) - 1.0)
        self.unit_delta_t = 1.0 / (self.grid_sizes_t.to(dtype) - 1.0)

    @staticmethod
    def _reflect_unit_cube(x_raw: torch.Tensor):
        return 1.0 - torch.abs((x_raw % 2.0) - 1.0)

    def snap_real_to_grid(self, x_real: torch.Tensor) -> torch.Tensor:
        idx = torch.round((x_real - self.lb_t) / self.delta_t)
        idx = torch.maximum(idx, torch.zeros_like(idx))
        idx = torch.minimum(idx, (self.grid_sizes_t - 1).to(device=idx.device, dtype=idx.dtype))
        x_snap = self.lb_t + idx.to(self.delta_t.dtype) * self.delta_t
        return x_snap

    def snap_unit_to_grid(self, x_unit: torch.Tensor) -> torch.Tensor:
        idx = torch.round(x_unit / self.unit_delta_t)
        idx = torch.maximum(idx, torch.zeros_like(idx))
        idx = torch.minimum(idx, (self.grid_sizes_t - 1).to(device=idx.device, dtype=idx.dtype))
        x_snap = idx.to(self.unit_delta_t.dtype) * self.unit_delta_t
        return x_snap

    def unit_to_grid_index(self, x_unit: torch.Tensor) -> torch.Tensor:
        idx = torch.round(x_unit / self.unit_delta_t)
        idx = torch.maximum(idx, torch.zeros_like(idx))
        idx = torch.minimum(idx, (self.grid_sizes_t - 1).to(device=idx.device, dtype=idx.dtype))
        return idx.to(torch.long)

    def sample_grid_prior_unit(self, n_particles: int, device, dtype) -> torch.Tensor:
        cols = []
        for j in range(self.nb_var):
            gj = int(self.grid_sizes_np[j])

            if self.prior_probs is None:
                idx_j = torch.randint(
                    low=0,
                    high=gj,
                    size=(n_particles,),
                    device=device,
                    dtype=torch.long,
                )
            else:
                probs_j = self.prior_probs[j]
                if probs_j is None:
                    idx_j = torch.randint(
                        low=0,
                        high=gj,
                        size=(n_particles,),
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    probs_j = torch.tensor(probs_j, device=device, dtype=dtype)
                    if probs_j.numel() != gj:
                        raise AssertionError(
                            f"prior_probs[{j}] should have length {gj}, got {probs_j.numel()}"
                        )
                    probs_j = probs_j / probs_j.sum()
                    idx_j = torch.multinomial(
                        probs_j,
                        num_samples=n_particles,
                        replacement=True,
                    )

            xj = idx_j.to(dtype) / float(gj - 1)
            cols.append(xj.unsqueeze(1))

        return torch.cat(cols, dim=1)

    @staticmethod
    def compute_ess_per_chain(chains: np.ndarray, max_lag: int | None = None):
        T, M = chains.shape
        if T < 5:
            return np.ones(M)

        if max_lag is None:
            max_lag = min(T // 2, 100)
        chains = np.asarray(chains, dtype=np.float64)

        if njit is not None:
            return _compute_ess_per_chain_numba(chains, max_lag)
        return _compute_ess_per_chain_numpy(chains, max_lag)

    @staticmethod
    def _rank_normalize(chains: np.ndarray):
        if chains.size == 0:
            return chains.astype(np.float64)

        flat = chains.reshape(-1)
        ranks = rankdata(flat, method="average")
        cdf = (ranks - 0.375) / (flat.size + 0.25)
        cdf = np.clip(cdf, 1e-12, 1.0 - 1e-12)
        z = norm.ppf(cdf)
        return z.reshape(chains.shape)

    @staticmethod
    def _classic_rhat(chains: np.ndarray):
        T, M = chains.shape
        if T < 2 or M < 2:
            return np.nan

        chain_means = np.mean(chains, axis=0)
        B = T * np.var(chain_means, ddof=1)
        W = np.mean(np.var(chains, axis=0, ddof=1))
        if not np.isfinite(B) or not np.isfinite(W) or W <= 1e-12:
            return np.nan

        var_hat = ((T - 1) / T) * W + (B / T)
        if var_hat <= 1e-12:
            return np.nan
        return float(np.sqrt(var_hat / W))

    @classmethod
    def compute_single_chain_rhat(cls, chain: np.ndarray):
        chain = np.asarray(chain, dtype=np.float64)
        T = chain.shape[0]
        half = T // 2
        if half < 10:
            return np.nan

        split = np.stack((chain[:half], chain[T - half:]), axis=1)
        return cls._classic_rhat(cls._rank_normalize(split))

    @staticmethod
    def _build_cert_diag_histories(prev_failed_histories, trace_buffer, n_cert, max_window=200):
        trace_arr = None
        if trace_buffer is not None and len(trace_buffer) > 0:
            trace_arr = np.stack(
                [
                    step.detach().cpu().numpy() if isinstance(step, torch.Tensor) else np.asarray(step)
                    for step in trace_buffer
                ],
                axis=0,
            )

        histories = []
        for chain_id in range(n_cert):
            prev_hist = prev_failed_histories[chain_id]
            if trace_arr is None:
                histories.append(prev_hist)
                continue

            chain_steps = trace_arr[:, chain_id]
            if chain_steps.ndim == 1:
                valid_mask = ~np.isnan(chain_steps)
                new_hist = chain_steps[valid_mask]
            else:
                valid_mask = ~np.any(np.isnan(chain_steps), axis=1)
                new_hist = chain_steps[valid_mask]

            if prev_hist.shape[0] == 0:
                chain_hist = new_hist.copy()
            elif new_hist.shape[0] == 0:
                chain_hist = prev_hist
            else:
                chain_hist = np.concatenate((prev_hist, new_hist), axis=0)
            histories.append(chain_hist)
        return histories

    def _propose_cert_subset(self, x: torch.Tensor, idx_subset: torch.Tensor, width_proposal: torch.Tensor):
        if idx_subset.numel() == 0:
            return x[idx_subset]

        if self.cert_use_cm:
            if self.nb_var > 10:
                cr = 1.0 / np.sqrt(float(self.nb_var))
                cross_mask = (
                    torch.rand(idx_subset.numel(), self.nb_var, device=x.device) < cr
                ).to(dtype=x.dtype)
                force_update_idx = torch.randint(
                    0, self.nb_var, (idx_subset.numel(), 1), device=x.device
                )
                cross_mask.scatter_(1, force_update_idx, 1.0)
            else:
                cross_mask = torch.ones(
                    (idx_subset.numel(), self.nb_var),
                    device=x.device,
                    dtype=x.dtype,
                )

            rw_step_cert = torch.randn_like(x[idx_subset]) * width_proposal[idx_subset]
            rw_step_cert = rw_step_cert * cross_mask
        else:
            rw_step_cert = torch.randn_like(x[idx_subset]) * width_proposal[idx_subset]

        x_maybe_cert_raw = x[idx_subset] + rw_step_cert
        x_maybe_cert_reflect = self._reflect_unit_cube(x_maybe_cert_raw)
        if self.snap_cert_pool:
            return self.snap_unit_to_grid(x_maybe_cert_reflect)
        return x_maybe_cert_reflect

    def _propose_search_subset(
        self,
        x: torch.Tensor,
        idx_subset: torch.Tensor,
        width_proposal: torch.Tensor,
        device,
        dtype,
        min_value=1.0,
    ):
        if idx_subset.numel() == 0:
            return x[idx_subset]

        if self.nb_var > 0 and idx_subset.numel() >= 3:
            local_perm = torch.randperm(idx_subset.numel(), device=device)
            idx1 = idx_subset[local_perm]
            idx2 = idx_subset[torch.roll(local_perm, 1)]

            use_de = (torch.rand(idx_subset.numel(), 1, device=device) < self.de_ratio).to(dtype)
            cr = 0.3 + 0.7 * torch.rand(idx_subset.numel(), 1, device=device)
            cross_mask = (torch.rand(idx_subset.numel(), self.nb_var, device=device) < cr).to(dtype)
            force_update_idx = torch.randint(0, self.nb_var, (idx_subset.numel(), 1), device=device)
            cross_mask.scatter_(1, force_update_idx, 1.0)

            d_eff = torch.clamp(cross_mask.sum(dim=1, keepdim=True), min=1.0)
            gamma_base = 2.38 / np.sqrt(2.0 * self.nb_var)
            gamma_dynamic = 2.38 / torch.sqrt(2.0 * d_eff)
            gamma_dynamic = torch.clamp(gamma_dynamic, min=gamma_base, max=2.0 * gamma_base)

            de_noise = (min_value * self.unit_delta_t).unsqueeze(0) * torch.randn_like(x[idx_subset])
            de_step = gamma_dynamic * (x[idx1] - x[idx2]) + de_noise
            rw_step = torch.randn_like(x[idx_subset]) * width_proposal[idx_subset]
            step = (use_de * de_step + (1.0 - use_de) * rw_step) * cross_mask
        else:
            step = torch.randn_like(x[idx_subset]) * width_proposal[idx_subset]

        x_maybe_search_raw = x[idx_subset] + step
        x_maybe_search_reflect = self._reflect_unit_cube(x_maybe_search_raw)
        if self.snap_search_pool:
            return self.snap_unit_to_grid(x_maybe_search_reflect)
        return x_maybe_search_reflect

        
    def _cert_diag_from_chain_histories(self, y_chain_histories, x_chain_histories):
        n_cert = len(y_chain_histories)
        if n_cert == 0:
            return {
                "collapsed_y": False,
                "rhat_max": np.nan,
                "ess_mean": np.inf,
                "ess_min": np.inf,
                "ess_max": np.inf,
                "pass_mask": np.empty(0, dtype=bool),
            }

        rhats = np.full(n_cert, np.nan, dtype=np.float64)
        ess_values = np.ones(n_cert, dtype=np.float64)

        for chain_id in range(n_cert):
            y_chain_hist = y_chain_histories[chain_id]
            Ty = len(y_chain_hist)
            if Ty >= self.cert_diag_max_window:
                rhats[chain_id] = 0.0
                ess_values[chain_id] = float(self.cert_diag_max_window)
                continue

            if Ty > 0:
                y_chain_arr = np.asarray(y_chain_hist, dtype=np.float64)
                if Ty >= 20:
                    rhats[chain_id] = self.compute_single_chain_rhat(y_chain_arr)

            x_chain_hist = x_chain_histories[chain_id]
            Tx = len(x_chain_hist)
            if Tx > 0:
                x_chain_arr = np.asarray(x_chain_hist, dtype=np.float64)
                if x_chain_arr.ndim == 1:
                    x_chain_arr = x_chain_arr[:, None]
                best_ess = 1.0
                for dim_id in range(x_chain_arr.shape[1]):
                    ess_dim = self.compute_ess_per_chain(
                        x_chain_arr[:, dim_id].reshape(-1, 1)
                    )[0]
                    if np.isfinite(ess_dim):
                        best_ess = max(best_ess, float(ess_dim))
                        if ess_dim >= self.ess:
                            break
                ess_values[chain_id] = best_ess

        finite_rhats = rhats[np.isfinite(rhats)]
        rhat_max = float(np.nanmax(finite_rhats)) if finite_rhats.size > 0 else np.nan
        ess_mean = float(np.nanmean(ess_values))
        ess_min = float(np.nanmin(ess_values))
        ess_max = float(np.nanmax(ess_values))
        collapsed = np.zeros(n_cert, dtype=bool)
        for chain_id in range(n_cert):
            y_chain_hist = y_chain_histories[chain_id]
            Ty = len(y_chain_hist)
            if Ty < 2:
                collapsed[chain_id] = False
                continue
            window = min(Ty, self.cert_diag_max_window)
            recent_y = np.asarray(y_chain_hist[-window:], dtype=np.float64)
            collapsed[chain_id] = np.var(recent_y, ddof=1) < 1e-12
        collapsed_y = bool(np.all(collapsed))

        pass_mask = (
            np.isfinite(rhats)
            & (rhats < 1.01)
            & np.isfinite(ess_values)
            & (ess_values >= self.ess)
        )

        return {
            "collapsed_y": collapsed_y,
            "rhat_max": rhat_max,
            "ess_mean": float(ess_mean),
            "ess_min": float(ess_min),
            "ess_max": float(ess_max),
            "pass_mask": pass_mask,
        }

    def _search_temperature(self, outer_iter: int, inner_iter: int):
        anneal_step = max(0, outer_iter * self.count_mh_steps + inner_iter)
        temp = self.search_sa_t0 * (self.search_sa_decay ** anneal_step)
        return max(self.search_sa_tmin, temp)

    def solve(self):
        count_particles = self.count_particles
        count_mh_steps = self.count_mh_steps
        min_value = 1

        min_width_cert = min_value / (self.grid_sizes - 1)
        min_width_search = min_value / (self.grid_sizes - 1)

        # --- split pools ---
        n_cert = max(2, int(round(count_particles * self.cert_fraction)))
        n_cert = min(n_cert, count_particles)
        n_search = count_particles - n_cert

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dtype = torch.float32
        dtype = self.bounds[0][0].dtype

        cert_idx = torch.arange(n_cert, device=device)
        search_idx = torch.arange(n_cert, count_particles, device=device)

        print(f"RW cert chains: {n_cert}, DE search chains: {n_search}")

        count_forward = 0

        self.lb_t = torch.tensor(self.lb, device=device, dtype=dtype)
        self.ub_t = torch.tensor(self.ub, device=device, dtype=dtype)
        self.range_t = self.ub_t - self.lb_t
        self._build_grid_info(device=device, dtype=dtype)

        # initial state on grid
        x = self.sample_grid_prior_unit(
            n_particles=count_particles,
            device=device,
            dtype=dtype,
        )

        width_proposal = torch.empty((count_particles, self.nb_var), device=device, dtype=dtype)
        width_proposal[cert_idx] = self.init_width_cert
        if n_search > 0:
            width_proposal[search_idx] = self.init_width_search

        # full population score history
        Y_all = torch.zeros(0, count_particles, device=device, dtype=dtype) * (-np.inf)

        # initial score
        x_real = self.lb_t + x * self.range_t
        s_x = -self.problem(x_real)
        if s_x.dim() > 1:
            s_x = s_x.squeeze(-1)
        count_forward += len(x)

        Y_all = torch.cat((Y_all, s_x.unsqueeze(0)), dim=0)

        max_n = 0

        log_p_point = 0.0
        log_p_pos = 0.0
        log_p_point_star_true = 0.0
        Y_all_n_plus1 = s_x.clone()

        max_value = torch.max(Y_all_n_plus1)
        if max_value > self.best_y:
            self.best_y = max_value
            best_idx = torch.argmax(Y_all_n_plus1)
            self.best_x = (self.lb_t + x[best_idx] * self.range_t).clone()

        cert_accept_y = [[] for _ in range(n_cert)]
        collapsed_y_count = 0
        y_all_flatten_buffer = []
        chain_id_flatten_buffer = []
        eval_cache = None
        if self.use_eval_cache and self.snap_cert_pool and self.snap_search_pool:
            eval_cache = {}
        cert_diag_y_history = [np.empty((0,), dtype=np.float64) for _ in range(n_cert)]
        cert_diag_x_history = [
            np.empty((0, self.nb_var), dtype=np.float64) for _ in range(n_cert)
        ]
        while torch.min(Y_all[-1][cert_idx]) < self.threshold:
            if self.best_y >= self.threshold and self.verify:
                break
            if (time.time() - self.start_time) > self.timeout_threshold:
                break

            # expand history
            Y_all = torch.cat((Y_all, Y_all[-1].unsqueeze(0)), dim=0)
            where_kill = Y_all[-1] < self.threshold
            cert_where_kill = where_kill[cert_idx]
            cert_count_kill = cert_where_kill.sum().item()
            if cert_count_kill == 0:
                break

            Y_all_n = Y_all[max_n].clone()
            Y_all_n_plus1 = Y_all[max_n + 1].clone()
            cert_active_idx = cert_idx
            cert_continuing_diag_mask = np.asarray(
                [hist.shape[0] > 0 for hist in cert_diag_y_history],
                dtype=bool,
            )
            y_trace_buffer = []
            x_trace_buffer = []
            total_acc_count = torch.zeros(count_particles, device=device, dtype=dtype)
            mh_executed = 0

            # --- MH loop with early stabilization checks ---
            for mh_idx in range(count_mh_steps):
                x_maybe = x.clone()

                # 1) RW-only cert pool
                if cert_active_idx.numel() > 0:
                    x_maybe[cert_active_idx] = self._propose_cert_subset(
                        x, cert_active_idx, width_proposal
                    )


                # 2) DE+RW search pool
                if n_search > 0:
                    x_maybe[search_idx] = self._propose_search_subset(
                        x,
                        search_idx,
                        width_proposal,
                        device,
                        dtype,
                        min_value=min_value,
                    )

                proposal_changed = (x_maybe != x).any(dim=1)
                if cert_active_idx.numel() > 0:
                    cert_need_candidates = cert_active_idx[where_kill[cert_active_idx]]
                    cert_unchanged = cert_need_candidates[~proposal_changed[cert_need_candidates]]
                    while cert_unchanged.numel() > 0:
                        x_maybe[cert_unchanged] = self._propose_cert_subset(
                            x, cert_unchanged, width_proposal
                        )
                        proposal_changed[cert_unchanged] = (x_maybe[cert_unchanged] != x[cert_unchanged]).any(dim=1)
                        cert_unchanged = cert_unchanged[~proposal_changed[cert_unchanged]]
                    
                if n_search > 0:
                    search_need_candidates = search_idx[where_kill[search_idx]]
                    search_unchanged = search_need_candidates[~proposal_changed[search_need_candidates]]
                    while (
                        search_unchanged.numel() > 0
                    ):
                        x_maybe[search_unchanged] = self._propose_search_subset(
                            x, search_unchanged, width_proposal, device, dtype
                        )
                        proposal_changed[search_unchanged] = (
                            x_maybe[search_unchanged] != x[search_unchanged]
                        ).any(dim=1)
                        search_unchanged = search_unchanged[~proposal_changed[search_unchanged]]

                # evaluate score
                s_maybe = Y_all_n_plus1.clone()
                if n_search > 0:
                    inactive_search_accept = search_idx[~where_kill[search_idx]]
                    if inactive_search_accept.numel() > 0:
                        x_maybe[inactive_search_accept] = x[inactive_search_accept]
                        proposal_changed[inactive_search_accept] = False
                need_eval = where_kill & proposal_changed

                if need_eval.any():
                    need_eval_idx = torch.where(need_eval)[0]

                    if eval_cache is None:
                        x_real_mcmc = self.lb_t + x_maybe[need_eval] * self.range_t

                        s_maybe_eval = -self.problem(x_real_mcmc)
                        if s_maybe_eval.dim() > 1:
                            s_maybe_eval = s_maybe_eval.squeeze(-1)
                        s_maybe[need_eval] = s_maybe_eval
                        count_forward += need_eval.sum().item()
                    else:
                        grid_idx_np = (
                            self.unit_to_grid_index(x_maybe[need_eval])
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        miss_positions = []

                        for local_pos, grid_idx_row in enumerate(grid_idx_np):
                            key = tuple(int(v) for v in grid_idx_row)
                            cached_score = eval_cache.get(key)
                            if cached_score is None:
                                miss_positions.append(local_pos)
                            else:
                                s_maybe[need_eval_idx[local_pos]] = cached_score

                        if miss_positions:
                            miss_pos_t = torch.tensor(
                                miss_positions, device=device, dtype=torch.long
                            )
                            miss_idx = need_eval_idx[miss_pos_t]
                            x_real_mcmc = self.lb_t + x_maybe[miss_idx] * self.range_t

                            s_maybe_eval = -self.problem(x_real_mcmc)
                            if s_maybe_eval.dim() > 1:
                                s_maybe_eval = s_maybe_eval.squeeze(-1)
                            s_maybe[miss_idx] = s_maybe_eval
                            count_forward += miss_idx.numel()

                            s_maybe_eval_np = (
                                s_maybe_eval.detach().cpu().numpy().astype(np.float64, copy=False)
                            )
                            for local_pos, score in zip(miss_positions, s_maybe_eval_np):
                                key = tuple(int(v) for v in grid_idx_np[local_pos])
                                eval_cache[key] = float(score)

                # acceptance
                acc_idx = torch.zeros_like(Y_all_n, dtype=torch.bool)
                acc_idx[cert_active_idx] = proposal_changed[cert_active_idx] & (
                    s_maybe[cert_active_idx] >= Y_all_n[cert_active_idx]
                )

                search_delta = s_maybe[search_idx] - Y_all_n_plus1[search_idx]
                search_floor_ok = s_maybe[search_idx] >= Y_all_n[search_idx]
                accept_search = search_delta >= 0
                if self.use_search_sa:
                    downhill_mask = ~accept_search
                    if downhill_mask.any():
                        temp = self._search_temperature(max_n, mh_idx)
                        accept_prob = torch.exp(
                            (search_delta[downhill_mask] / temp).clamp(min=-60.0, max=0.0)
                        )
                        accept_search[downhill_mask] = (
                            torch.rand_like(search_delta[downhill_mask]) < accept_prob
                        )
                acc_idx[search_idx] = proposal_changed[search_idx] & search_floor_ok & accept_search

                x = torch.where(acc_idx.unsqueeze(-1), x_maybe, x)
                Y_all_n_plus1 = torch.where(acc_idx, s_maybe.detach(), Y_all_n_plus1)

                total_acc_count += acc_idx.float()
                mh_executed = mh_idx + 1

                # cert diagnostics only: collect after burn-in
                should_collect = ((mh_idx + 1) % self.cert_diag_stride == 0)
                if should_collect:
                    y_step = Y_all_n_plus1[cert_active_idx].detach().cpu().numpy()
                    x_step = x[cert_active_idx].detach().cpu().numpy()
                    if mh_idx >= self.cert_diag_burn_in:
                        y_trace_buffer.append(y_step)
                        x_trace_buffer.append(x_step)
                    elif np.any(cert_continuing_diag_mask):
                        y_masked = np.full_like(y_step, np.nan)
                        x_masked = np.full_like(x_step, np.nan)
                        y_masked[cert_continuing_diag_mask] = y_step[cert_continuing_diag_mask]
                        x_masked[cert_continuing_diag_mask] = x_step[cert_continuing_diag_mask]
                        y_trace_buffer.append(y_masked)
                        x_trace_buffer.append(x_masked)
                # best global counterexample
                max_value = torch.max(Y_all_n_plus1)
                if max_value > self.best_y:
                    self.best_y = max_value
                    best_idx = torch.argmax(Y_all_n_plus1)
                    self.best_x = (self.lb_t + x[best_idx] * self.range_t).clone()



                # search success inside MH loop
                if self.best_y >= self.threshold and self.verify:
                    break

            final_acc_ratio = total_acc_count / max(mh_executed, 1)
            zero_accept_cert_outer = final_acc_ratio[cert_idx] <= 0.0

            # final diagnostics on cert pool after MH loop
            cert_chain_y_histories = self._build_cert_diag_histories(
                cert_diag_y_history,
                y_trace_buffer,
                cert_active_idx.numel(),
                max_window=self.cert_diag_max_window,
            )
            cert_chain_x_histories = self._build_cert_diag_histories(
                cert_diag_x_history,
                x_trace_buffer,
                cert_active_idx.numel(),
                max_window=self.cert_diag_max_window,
            )
            diag = self._cert_diag_from_chain_histories(
                cert_chain_y_histories,
                cert_chain_x_histories,
            )
            zero_accept_cert_outer_np = zero_accept_cert_outer.detach().cpu().numpy()
            cert_diag_pass_mask_np = np.asarray(diag["pass_mask"], dtype=bool)
            cert_diag_pass_mask_np = cert_diag_pass_mask_np & (~zero_accept_cert_outer_np)
            cert_diag_pass_count = int(np.sum(cert_diag_pass_mask_np))
            collapsed_y = diag["collapsed_y"]
            if collapsed_y:
                collapsed_y_count += 1
            else:
                collapsed_y_count = 0
            rhat_max = diag["rhat_max"]
            ess_mean = diag["ess_mean"]
            ess_min = diag["ess_min"]
            ess_max = diag["ess_max"]

            passed_cert_outer = torch.tensor(
                cert_diag_pass_mask_np,
                device=device,
                dtype=torch.bool,
            )
            passed_cert_outer_np = passed_cert_outer.detach().cpu().numpy()
            failed_cert_outer = ~passed_cert_outer
            if failed_cert_outer.any():
                failed_cert_global = cert_idx[failed_cert_outer]
                Y_all_n_plus1[failed_cert_global] = Y_all_n[failed_cert_global]

            valid_cert_sample_outer = passed_cert_outer & (Y_all_n_plus1[cert_idx] < self.threshold)
            if valid_cert_sample_outer.any():
                passed_cert_scores = (
                    Y_all_n_plus1[cert_idx][valid_cert_sample_outer].detach().cpu().tolist()
                )
                passed_cert_ids = torch.where(valid_cert_sample_outer)[0].detach().cpu().tolist()
                for chain_local, score in zip(passed_cert_ids, passed_cert_scores):
                    cert_accept_y[int(chain_local)].append(float(score))
                    y_all_flatten_buffer.append(float(score))
                    chain_id_flatten_buffer.append(int(chain_local))

            for chain_local in range(cert_active_idx.numel()):
                if passed_cert_outer_np[chain_local]:
                    cert_diag_y_history[chain_local] = np.empty((0,), dtype=np.float64)
                    cert_diag_x_history[chain_local] = np.empty(
                        (0, self.nb_var), dtype=np.float64
                    )
                else:
                    cert_diag_y_history[chain_local] = cert_chain_y_histories[chain_local]
                    cert_diag_x_history[chain_local] = cert_chain_x_histories[chain_local]

            # width adaptation
            target_cert = 0.234
            target_search = 0.234
            eta = 0.1

            logw = torch.log(width_proposal.clamp(min=min_width_cert))

            if n_cert > 0:
                adj_cert = eta * (final_acc_ratio[cert_idx] - target_cert)
                # passed_cert_idx = cert_idx[cert_diag_pass_mask_np]
                # if passed_cert_idx.numel() > 0:
                    # logw[passed_cert_idx] = logw[passed_cert_idx] + adj_cert[cert_diag_pass_mask_np].unsqueeze(-1)
                logw[cert_idx] = logw[cert_idx] + adj_cert.unsqueeze(-1)

            if n_search > 0:
                adj_search = eta * (final_acc_ratio[search_idx] - target_search)
                logw[search_idx] = logw[search_idx] + adj_search.unsqueeze(-1)

            width_proposal = torch.exp(logw).clamp(min=min_width_search, max=1.0)
            
            Y_all[max_n + 1] = Y_all_n_plus1.clone()
            max_n += 1

            Y_all_flatten = np.asarray(y_all_flatten_buffer, dtype=np.float64)
            chain_id_flatten = np.asarray(chain_id_flatten_buffer, dtype=np.int64)
            longest_chain_depth = max((len(v) for v in cert_accept_y), default=0)
            log_p_pos = np.log((1 - 1 / n_cert)) * int(Y_all_flatten.size)

            alpha = 1e-5
            alpha_t = alpha * (6.0 / (np.pi**2)) / (max_n**2)
            res = repp_upper_bound(
                Y_all_flatten,
                chain_id_flatten,
                m_total=int(Y_all_flatten.size),
                N=n_cert,
                alpha=alpha_t,
            )
            log_p_point_star_true = res["logU"]
            log_p_point = res["log_p_point"]


            if max_n % 10 == 0:
                # print(cert_stalled , bound_stable , (not best_y_progressing) , (not near_threshold))
                if n_search > 0:
                    print(
                        f"[iter {max_n} - "
                        f"{(Y_all[-1][search_idx] >= self.threshold).sum()} search crossed, "
                        f"{(Y_all[-1][cert_idx] >= self.threshold).sum()} cert crossed]: "
                        f"[log_p_pos = {log_p_pos:.2f}, "
                        f"log_p_point = {log_p_point:.2f}, "
                        f"log_p_upper = {log_p_point_star_true:.2f}] "
                        f"[cert_diag_pass={cert_diag_pass_count}/{cert_active_idx.numel()} "
                        f"rhat_max={rhat_max:.3f}, "
                        f"ess_mean={ess_mean:.2f}]"
                        f"cert=[{-torch.max(Y_all[-1][cert_idx]):.5f} {-torch.min(Y_all[-1][cert_idx]):.5f} {final_acc_ratio[cert_idx].mean().item():.3f}] search=[{-torch.max(Y_all[-1][search_idx]):.5f} {-torch.min(Y_all[-1][search_idx]):.5f} {final_acc_ratio[search_idx].mean().item():.3f}] "
                        f"result=[y={-self.best_y:.5f} K={count_forward} time={(time.time() - self.start_time):2f} chain_length={longest_chain_depth} collapsed_y={collapsed_y} collapsed_count={collapsed_y_count}]",
                        flush=True,
                    )
                else:
                    print(
                        f"[iter {max_n} - "
                        f"{(Y_all[-1][cert_idx] >= self.threshold).sum()} cert crossed]: "
                        f"[log_p_pos = {log_p_pos:.2f}, "
                        f"log_p_point = {log_p_point:.2f}, "
                        f"log_p_upper = {log_p_point_star_true:.2f}] "
                        f"[cert_diag_pass={cert_diag_pass_count}/{cert_active_idx.numel()} "
                        f"rhat_max={rhat_max:.3f} "
                        f"ess_mean={ess_mean:.2f}] "
                        f"cert=[{-torch.max(Y_all[-1][cert_idx]):.5f} {-torch.min(Y_all[-1][cert_idx]):.5f} {final_acc_ratio[cert_idx].mean().item():.3f}] "
                        f"result=[y={-self.best_y:.5f} K={count_forward} time={(time.time() - self.start_time):2f} chain_length={longest_chain_depth} collapsed_y={collapsed_y} collapsed_count={collapsed_y_count}]",
                        flush=True,
                    )


            self.repp = log_p_point
            self.query = count_forward
            self.log_p_point_star_true = log_p_point_star_true

            # 1) search success
            if self.best_y >= self.threshold and self.verify:
                if n_search > 0:
                    print(
                        f"Success!"
                        f"{(Y_all[-1][search_idx] >= self.threshold).sum()} search particles crossed, "
                        f"{(Y_all[-1][cert_idx] >= self.threshold).sum()} cert particles crossed, "
                        f"log_p_point={log_p_point}, "
                        f"log_p_upper={log_p_point_star_true}",
                        flush=True,
                    )
                else:
                    print(
                        f"Success!"
                        f"{(Y_all[-1][cert_idx] >= self.threshold).sum()} cert particles crossed, "
                        f"log_p_point={log_p_point}, "
                        f"log_p_upper={log_p_point_star_true}",
                        flush=True,
                    )
                return log_p_pos, log_p_point, log_p_point_star_true, count_forward, x

            # 2) certified p <= tau on RW cert pool
            if log_p_point_star_true <= self.tau and cert_count_kill == n_cert:
                print(
                    f"Certified with tau={self.tau}. "
                    f"log_p_upper={log_p_point_star_true}",
                    flush=True,
                )
                return log_p_pos, log_p_point, log_p_point_star_true, count_forward, x

            if collapsed_y_count >= self.cert_no_growth_patience:
                self.log_p_point_star_true = self.tau -1
                print(
                    f"Early stop: cert y histories collapsed for "
                    f"{self.cert_no_growth_patience} consecutive outer iterations.",
                    flush=True,
                )
                return log_p_pos, log_p_point, log_p_point_star_true, count_forward, x


        max_value = torch.max(Y_all_n_plus1)
        if max_value > self.best_y:
            self.best_y = max_value
            best_idx = torch.argmax(Y_all_n_plus1)
            self.best_x = (self.lb_t + x[best_idx] * self.range_t).clone()

        print(
            f"Done. N_cert = {n_cert}, N_search = {n_search}, "
            f"log_p_point={log_p_point}, "
            f"log_p_upper={log_p_point_star_true}",
            flush=True,
        )
        self.repp = log_p_point
        self.query = count_forward
        self.log_p_point_star_true = log_p_point_star_true

        return log_p_pos, log_p_point, log_p_point_star_true, count_forward, x
