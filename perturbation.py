import torch

class Perturbation_Biasfield:
    def __init__(
        self,
        epsilon: float,
        order: int,
        clip_inputs: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.epsilon = float(epsilon)
        self.order = int(order)
        self.clip_inputs = bool(clip_inputs)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.device = device
        self.dtype = dtype

        self.num_coeff = (self.order + 1) ** 2
        self.bounds = self._build_bounds()

    @staticmethod
    def _power_series(x_: torch.Tensor, n_order: int) -> torch.Tensor:
        ones = torch.ones_like(x_)
        s_ = torch.stack([ones] + [x_] * n_order, dim=0)
        return torch.cumprod(s_, dim=0)

    @staticmethod
    def _get_meshgrid(num_points_x: int, num_points_y: int | None = None):
        if num_points_y is None:
            num_points_y = num_points_x
        x = torch.linspace(0, 1, num_points_x)
        y = torch.linspace(0, 1, num_points_y)
        x, y = torch.meshgrid(x, y, indexing="ij")
        return x.T, y.T

    @classmethod
    def _get_bias_field_matrix(cls, x: torch.Tensor, y: torch.Tensor, order: int) -> torch.Tensor:
        x_powers = cls._power_series(x, order)
        y_powers = cls._power_series(y, order)
        combos = [
            x_powers[k] * y_powers[j]
            for i in range(order, -1, -1)
            for k, j in [(i, j) for j in range(i)] + [(j, i) for j in range(i, -1, -1)]
        ]
        return torch.stack(combos, dim=-1)

    @classmethod
    def get_biasfields_4d(
        cls,
        image: torch.Tensor,
        order: int = 1,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"image must have shape [1,C,H,W], got {image.shape}")

        c = image.shape[1]
        h, w = image.shape[2:]

        x_coords, y_coords = cls._get_meshgrid(w, h)
        if device is not None:
            x_coords = x_coords.to(device)
            y_coords = y_coords.to(device)
        if dtype is not None:
            x_coords = x_coords.to(dtype=dtype)
            y_coords = y_coords.to(dtype=dtype)

        basis = cls._get_bias_field_matrix(x_coords, y_coords, order=order)
        basis = basis.reshape(1, h, w, (order + 1) ** 2)
        basis = basis.repeat(c, 1, 1, 1)
        basis = torch.permute(basis, (3, 0, 1, 2))
        return basis

    def _build_bounds(self):
        k = self.num_coeff
        device = self.device if self.device is not None else "cpu"

        bounds = torch.zeros(k, 2, device=device, dtype=self.dtype)
        lower, upper = bounds[:, 0], bounds[:, 1]

        if k > 1:
            lower[:-1] = -self.epsilon / (k - 1)
            upper[:-1] =  self.epsilon / (k - 1)

        lower[-1] = 1.0 / (1.0 + self.epsilon)
        upper[-1] = 1.0 + self.epsilon
        return bounds

    def nominal_coeff(self, device=None, dtype=None) -> torch.Tensor:
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        coeff = torch.zeros(self.num_coeff, device=device, dtype=dtype)
        coeff[-1] = 1.0
        return coeff

    def prepare(self, frames: torch.Tensor):
        if frames.ndim != 4:
            raise ValueError(f"Expected [T,C,H,W], got {frames.shape}")

        device = frames.device if self.device is None else self.device
        dtype = self.dtype

        frames = frames.to(device=device, dtype=dtype)

        if self.clip_max == 1:
            frames = frames / 255.0

        self.frames_prepared = frames
        first_frame = frames[0:1]
        self.basis = self.get_biasfields_4d(
            first_frame,
            order=self.order,
            device=device,
            dtype=dtype,
        )
        return self

    def transform_func(self, x, frames: torch.Tensor | None = None) -> torch.Tensor:
        if frames is None:
            if not hasattr(self, "frames_prepared") or not hasattr(self, "basis"):
                raise RuntimeError("Call prepare(frames) first, or pass frames explicitly.")
            frames = self.frames_prepared
            basis = self.basis
        else:
            device = frames.device if self.device is None else self.device
            dtype = self.dtype
            frames = frames.to(device=device, dtype=dtype)

            if self.clip_max == 1:
                frames = frames / 255.0

            first_frame = frames[0:1]
            basis = self.get_biasfields_4d(
                first_frame,
                order=self.order,
                device=device,
                dtype=dtype,
            )

        if not torch.is_tensor(x):
            coeff = torch.tensor(x, device=frames.device, dtype=frames.dtype)
        else:
            coeff = x.to(device=frames.device, dtype=frames.dtype)

        if coeff.ndim == 1:
            coeff = coeff.unsqueeze(0)
        elif coeff.ndim != 2:
            raise ValueError(f"coeff must have shape [K] or [B, K], got {coeff.shape}")

        coeff = coeff.contiguous()
        basis = basis.contiguous()
        frames = frames.contiguous()

        B = coeff.shape[0]
        K, C, H, W = basis.shape
        T = frames.shape[0]

        basis_flat = basis.view(K, -1)          # [K, C*H*W]
        field = coeff @ basis_flat              # [B, C*H*W]
        field = field.view(B, C, H, W)          # [B, C, H, W]

        perturbed = frames.unsqueeze(0) * field.unsqueeze(1)   # [B, T, C, H, W]

        if self.clip_inputs:
            perturbed = torch.clamp(perturbed, min=self.clip_min, max=self.clip_max)

        if self.clip_max == 1:
            perturbed = perturbed * 255.0

        return perturbed
