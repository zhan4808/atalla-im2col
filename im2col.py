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
    use_sequential_init: bool = True


class ImplicitIm2colSystolicSim:
    """
    Channel-first implicit im2col simulation with systolic-array mapping.
    HWC layout: each input "word" is all channels for one pixel (h, w).
    Each (r, s) kernel position is a CxK 1x1 GEMM.
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

    def explicit_im2col_channel_first(self):
        cfg = self.cfg
        padded = self._pad_ifmap()
        rows = []
        for n in range(cfg.N):
            for oh in range(self.Ho):
                for ow in range(self.Wo):
                    cols = []
                    for r in range(cfg.R):
                        for s in range(cfg.S):
                            ih = oh * cfg.stride + r
                            iw = ow * cfg.stride + s
                            cols.extend(padded[n, ih, iw, :].tolist())
                    rows.append(cols)
        return np.array(rows)

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

    def simulate(self, trace=True):
        cfg = self.cfg
        padded_ifmap = self._pad_ifmap()
        logs = []

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

                            if trace:
                                logs.append(
                                    {
                                        "tile_rs": (r, s),
                                        "n": n,
                                        "out_hw": (oh, ow),
                                        "in_hw": (ih, iw),
                                        "weight_tile": wt[:cfg.C, :cfg.K].tolist(),
                                        "input_word": input_vec[:cfg.C].tolist(),
                                        "partial_sum": partial.tolist(),
                                    }
                                )

        return self.ofmap, logs

    def simulate_systolic(self, trace=True):
        """
        Functional systolic timing:
        - Row i input arrives at time i (skewed injection).
        - Output becomes ready after (cin_limit - 1) + (array_size - 1).
        This does not model memory or exact cycle counts, only dataflow order.
        """
        cfg = self.cfg
        self.ofmap = np.zeros((cfg.N, self.Ho, self.Wo, cfg.K), dtype=float)
        padded_ifmap = self._pad_ifmap()
        logs = []

        cin_limit = min(cfg.C, cfg.array_size)
        kout_limit = min(cfg.K, cfg.array_size)
        tiles_per_pack = max(1, min(cfg.array_size // max(cfg.C, 1), cfg.S))

        for r in range(cfg.R):
            for s_base in range(0, cfg.S, tiles_per_pack):
                group_s = list(range(s_base, min(s_base + tiles_per_pack, cfg.S)))
                packed_weight = np.zeros((cfg.array_size, cfg.array_size), dtype=float)

                for t_idx, s in enumerate(group_s):
                    row_base = t_idx * cfg.C
                    if row_base >= cfg.array_size:
                        break
                    packed_weight[row_base:row_base + cin_limit, :kout_limit] = self.weights[
                        r, s, :cin_limit, :kout_limit
                    ]

                for n in range(cfg.N):
                    for oh in range(self.Ho):
                        for ow in range(self.Wo):
                            packed_input = np.zeros((cfg.array_size,), dtype=float)
                            tile_partials = []
                            tile_inputs = []
                            tile_weights = []

                            for t_idx, s in enumerate(group_s):
                                row_base = t_idx * cfg.C
                                if row_base >= cfg.array_size:
                                    break
                                ih = oh * cfg.stride + r
                                iw = ow * cfg.stride + s
                                input_vec = self.input_word(padded_ifmap, n, ih, iw)
                                packed_input[row_base:row_base + cin_limit] = input_vec[:cin_limit]

                                wt = self.weights[r, s, :cin_limit, :kout_limit]
                                partial = input_vec[:cin_limit] @ wt
                                self.ofmap[n, oh, ow, :kout_limit] += partial

                                tile_partials.append(partial.tolist())
                                tile_inputs.append(input_vec[:cfg.C].tolist())
                                tile_weights.append(wt[:cfg.C, :cfg.K].tolist())

                            timeline = []
                            if trace:
                                running = [np.zeros((kout_limit,), dtype=float) for _ in tile_partials]
                                for t in range(cfg.array_size):
                                    row_inputs = [0.0] * cfg.array_size
                                    if t < cfg.array_size:
                                        row_inputs[t] = float(packed_input[t])

                                    t_idx = t // cfg.C
                                    row_in_tile = t - (t_idx * cfg.C)
                                    if t_idx < len(tile_partials) and row_in_tile < cin_limit:
                                        w_row = packed_weight[t, :kout_limit]
                                        running[t_idx] += packed_input[t] * w_row

                                    timeline.append(
                                        {
                                            "t": t,
                                            "row_inputs": row_inputs[:cfg.array_size],
                                            "partial_sums": [r.tolist() for r in running],
                                        }
                                    )

                                logs.append(
                                    {
                                        "tile_rs_group": [(r, s) for s in group_s],
                                        "n": n,
                                        "out_hw": (oh, ow),
                                        "weight_tiles": tile_weights,
                                        "input_words": tile_inputs,
                                        "packed_weight_tile": packed_weight[: cfg.array_size, : cfg.K].tolist(),
                                        "packed_input": packed_input[:cfg.array_size].tolist(),
                                        "partial_sums": tile_partials,
                                        "output_ready_time": (cin_limit - 1) + (cfg.array_size - 1),
                                        "timeline": timeline,
                                    }
                                )

        return self.ofmap, logs


def direct_conv_hwc(ifmap, weights, stride=1, pad=0):
    n, h, w, c = ifmap.shape
    r, s, c2, k = weights.shape
    if c != c2:
        raise ValueError("IFMap C and weight C mismatch.")
    ho = (h + 2 * pad - r) // stride + 1
    wo = (w + 2 * pad - s) // stride + 1
    padded = np.pad(ifmap, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    out = np.zeros((n, ho, wo, k), dtype=float)
    for nn in range(n):
        for oh in range(ho):
            for ow in range(wo):
                acc = np.zeros((k,), dtype=float)
                for rr in range(r):
                    for ss in range(s):
                        ih = oh * stride + rr
                        iw = ow * stride + ss
                        acc += padded[nn, ih, iw, :] @ weights[rr, ss, :, :]
                out[nn, oh, ow, :] = acc
    return out


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
        use_sequential_init=True,
    )
    sim = ImplicitIm2colSystolicSim(cfg)

    print("IFMap (HWC):")
    print(sim.ifmap)
    print("\nWeights (R,S,C,K):")
    print(sim.weights)

    print("\nUnfolded IFMap (channel-first order HF->WF->C):")
    unfolded = sim.explicit_im2col_channel_first()
    print(unfolded)

    ofmap, logs = sim.simulate_systolic(trace=True)
    print("\nIteration logs (systolic-timed, packed):")
    for i, entry in enumerate(logs):
        print(f"iter={i} tiles={entry['tile_rs_group']} out={entry['out_hw']}")
        for t_idx, tile in enumerate(entry["tile_rs_group"]):
            print(f"  tile={tile}")
            print(f"    weight_tile:\n{np.array(entry['weight_tiles'][t_idx])}")
            print(f"    input_word: {entry['input_words'][t_idx]}")
            print(f"    partial_sum: {entry['partial_sums'][t_idx]}")
        print(f"  packed_input: {entry['packed_input']}")
        print(f"  packed_weight_tile:\n{np.array(entry['packed_weight_tile'])}")
        print(f"  output_ready_time: {entry['output_ready_time']}")
        for step in entry["timeline"]:
            print(f"    t={step['t']} row_inputs={step['row_inputs']} partial_sums={step['partial_sums']}")

    print("\nFinal OFMap (HWC):")
    print(ofmap)

    ref = direct_conv_hwc(sim.ifmap, sim.weights, stride=cfg.stride, pad=cfg.pad)
    diff = np.max(np.abs(ref - ofmap))
    print("\nVerification (direct conv vs sim):")
    print(f"max_abs_diff={diff}")
    print("PASS" if diff < 1e-6 else "FAIL")


if __name__ == "__main__":
    main()