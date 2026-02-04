from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    N: int
    H: int
    W: int
    C: int
    K: int
    R: int
    S: int
    array_size: int
    stride: int = 1
    pad: int = 0
    seed: int = 0
    trace: bool = True
    max_trace: int = 64
    use_sequential_init: bool = True


class ImplicitIm2colSystolicSim:
    """
    Channel-first implicit im2col simulation with systolic-array mapping.
    HWC layout: each input "word" is all channels for one pixel (h, w).
    Weight tiles: each (r, s) is a CxK 1x1 GEMM.
    """

    def __init__(self, cfg: SimConfig, ifmap=None, weights=None):
        self.cfg = cfg
        self._validate()
        self.Ho = (cfg.H + 2 * cfg.pad - cfg.R) // cfg.stride + 1
        self.Wo = (cfg.W + 2 * cfg.pad - cfg.S) // cfg.stride + 1

        rng = np.random.default_rng(cfg.seed)
        if ifmap is None:
            if cfg.use_sequential_init:
                self.ifmap = np.arange(cfg.N * cfg.H * cfg.W * cfg.C).reshape(
                    cfg.N, cfg.H, cfg.W, cfg.C
                )
            else:
                self.ifmap = rng.integers(0, 10, (cfg.N, cfg.H, cfg.W, cfg.C))
        else:
            self.ifmap = np.array(ifmap, copy=True)

        if weights is None:
            if cfg.use_sequential_init:
                self.weights = np.arange(cfg.R * cfg.S * cfg.C * cfg.K).reshape(
                    cfg.R, cfg.S, cfg.C, cfg.K
                )
            else:
                self.weights = rng.integers(0, 10, (cfg.R, cfg.S, cfg.C, cfg.K))
        else:
            self.weights = np.array(weights, copy=True)

        self.ofmap = np.zeros((cfg.N, self.Ho, self.Wo, cfg.K), dtype=float)

    def _validate(self):
        cfg = self.cfg
        if cfg.array_size <= 0 or cfg.array_size != int(cfg.array_size):
            raise ValueError("array_size must be a positive integer.")
        if cfg.H + 2 * cfg.pad < cfg.R or cfg.W + 2 * cfg.pad < cfg.S:
            raise ValueError("Kernel larger than padded input.")
        if cfg.stride <= 0:
            raise ValueError("stride must be >= 1.")

    def _pad_ifmap(self):
        cfg = self.cfg
        return np.pad(
            self.ifmap,
            ((0, 0), (cfg.pad, cfg.pad), (cfg.pad, cfg.pad), (0, 0)),
            mode="constant",
        )

    def weight_tile(self, r, s):
        cfg = self.cfg
        tile = np.zeros((cfg.array_size, cfg.array_size), dtype=float)
        cin_limit = min(cfg.C, cfg.array_size)
        kout_limit = min(cfg.K, cfg.array_size)
        tile[:cin_limit, :kout_limit] = self.weights[r, s, :cin_limit, :kout_limit]
        return tile

    def input_word(self, padded_ifmap, n, ih, iw):
        cfg = self.cfg
        word = padded_ifmap[n, ih, iw, :]
        if word.shape[0] < cfg.array_size:
            padded = np.zeros((cfg.array_size,), dtype=word.dtype)
            padded[: word.shape[0]] = word
            return padded
        return word[: cfg.array_size]

    def tile_input_indices(self, r, s):
        cfg = self.cfg
        indices = []
        for oh in range(self.Ho):
            for ow in range(self.Wo):
                ih = oh * cfg.stride + r
                iw = ow * cfg.stride + s
                indices.append((oh, ow, ih, iw))
        return indices

    def simulate(self):
        cfg = self.cfg
        padded_ifmap = self._pad_ifmap()
        trace = []

        for r in range(cfg.R):
            for s in range(cfg.S):
                wt = self.weight_tile(r, s)
                cin_limit = min(cfg.C, cfg.array_size)
                kout_limit = min(cfg.K, cfg.array_size)

                for n in range(cfg.N):
                    for oh in range(self.Ho):
                        for ow in range(self.Wo):
                            ih = oh * cfg.stride + r
                            iw = ow * cfg.stride + s
                            input_vec = self.input_word(padded_ifmap, n, ih, iw)
                            partial = input_vec[:cin_limit] @ wt[:cin_limit, :kout_limit]
                            self.ofmap[n, oh, ow, :kout_limit] += partial

                            if cfg.trace and len(trace) < cfg.max_trace:
                                trace.append(
                                    {
                                        "tile_rs": (r, s),
                                        "n": n,
                                        "out_hw": (oh, ow),
                                        "in_hw": (ih, iw),
                                        "input_word": input_vec[:cfg.C].tolist(),
                                        "weight_slice": self.weights[r, s, :cin_limit, :kout_limit].tolist(),
                                        "partial_sum": partial.tolist(),
                                    }
                                )

        return self.ofmap, trace


def main():
    cfg = SimConfig(
        N=1,
        H=4,
        W=4,
        C=3,
        K=2,
        R=2,
        S=2,
        array_size=4,
        stride=1,
        pad=0,
        trace=True,
        max_trace=16,
        use_sequential_init=True,
    )
    sim = ImplicitIm2colSystolicSim(cfg)
    ofmap, trace = sim.simulate()

    print("IFMap (HWC):")
    print(sim.ifmap)
    print("\nWeights (R,S,C,K):")
    print(sim.weights)
    print("\nTrace (first entries):")
    for entry in trace:
        print(entry)
    print("\nOFMap (HWC):")
    print(ofmap)


if __name__ == "__main__":
    main()