下面給你兩個層級的 Mai-kernel 偽程式：
A.「精確但可向量化」版本（用行列式身分加速，無需四重迴圈）
B.「可擴展」版本（加入 RFF/Nyström、時間對齊抽樣）

---

# A) Mai-kernel（向量化精確版）

```pseudo
# ===== 資料結構 =====
# 一段影片 V 由 T 個時間步組成
# 於時間 t:
#   Z_t ∈ R^{N_t × d}     # 物件/人/手-物 的特徵向量
#   M_t ∈ [0,1]^{N_t}     # 可見遮罩(1=可見, 0=不可見; 亦可為機率)

# ===== 介面 =====
# MaiKernel(V, V_prime, params) -> scalar (≥ 0)
# params.object_kernel: {"type": "rbf"/"linear"/"poly", ...}
# params.time_kernel  : {"type": "gaussian"/"GAK"/"softDTW", ...}

function MaiKernel(V, V_prime, params):
    T  ← length(V)            # 影格數
    S  ← length(V_prime)
    W  ← TimeAlignmentWeights(T, S, params.time_kernel) 
         # W[t,s] = κ_T(t,s)，可用 Gaussian(time) 或 GAK/softDTW 產生平滑權重

    total ← 0.0

    for t in 1..T:
        Zt ← V.Z[t]           # (N_t × d)
        Mt ← V.M[t]           # (N_t)
        for s in 1..S:
            Zs ← V_prime.Z[s] # (N'_s × d)
            Ms ← V_prime.M[s] # (N'_s)

            # 1) 物件對物件的基底核矩陣 K = κ_Z(Zt, Zs) (N_t × N'_s)
            K ← ObjectKernelMatrix(Zt, Zs, params.object_kernel)

            # 2) 以遮罩門控：A = K ⊙ (Mt * Ms^T)
            #    其中 (Mt * Ms^T)[i,u] = Mt[i] * Ms[u]
            A ← ElementwiseMultiply(K, Outer(Mt, Ms))

            # 3) 互動對總和（i≠j, u≠v）的高效計算
            #    身分：sum_{i≠j, u≠v} A[i,u]A[j,v] 
            #        = (sum A)^2 - ||row_sums(A)||_2^2 - ||col_sums(A)||_2^2 + ||A||_F^2
            s_tot ← SumAll(A)                        # 標量
            r     ← RowSums(A)                       # (N_t)
            c     ← ColSums(A)                       # (N'_s)
            fro2  ← SumAll(Square(A))                # Frobenius norm^2

            pair_sum ← s_tot^2 - Dot(r, r) - Dot(c, c) + fro2

            # 4) 乘上時間對齊權重並累加
            total ← total + W[t,s] * pair_sum

    return total


# ----- 基底元件 -----
function ObjectKernelMatrix(X, Y, kparams):
    # X: (n×d), Y: (m×d)
    if kparams.type == "linear":
        return X @ Y^T
    else if kparams.type == "rbf":
        # K[i,u] = exp(-γ ||X[i]-Y[u]||^2)
        γ ← kparams.gamma
        return RBFKernel(X, Y, γ)
    else if kparams.type == "poly":
        # (α x⋅y + c)^p
        α, c, p ← kparams.alpha, kparams.c, kparams.degree
        return Power(α*(X @ Y^T) + c, p)
    else:
        raise NotImplemented

function TimeAlignmentWeights(T, S, tparams):
    if tparams.type == "gaussian":
        # W[t,s] = exp(- (t-s)^2 / (2σ^2))，再正規化使 sum_{t,s} W[t,s] = 1
        σ ← tparams.sigma
        W[t,s] = exp(- ((t-s)^2) / (2*σ^2))  for all t,s
        return W / SumAll(W)
    else if tparams.type == "GAK" or tparams.type == "softDTW":
        # 以 GAK 或 soft-DTW 產生對齊權重矩陣（可微、DP 求解）
        # 回傳非負矩陣 W，並正規化至 sum W = 1
        W ← AlignmentWeightsByDP(T, S, tparams)
        return W / SumAll(W)
    else:
        raise NotImplemented
```

**說明**

* 第 3 步的身分式讓你**不必**顯式枚舉所有互動對 $(i\neq j, u\neq v)$，每個時間對 $(t,s)$ 僅需 $O(N_t N'_s)$ 時間建立 $K$ 與簡單的向量化和平方運算即可。
* 遮罩 $M_t, M'_s$ 直接把看不到的物件歸零，等價於在 RKHS 內做「可見子集」的條件比對。
* $W$ 由時間核/對齊產生，會把相近或可對齊的時間片加大權重。

---

# B) Mai-kernel（可擴展版：RFF/Nyström + 時間抽樣）

```pseudo
# 可擴展參數
# params.rff: {use: bool, D: int, gamma: float, seed: int}
# params.nystrom: {use: bool, landmarks: int}
# params.time_sampling: {use: bool, K_pairs: int, scheme: "topk"/"proportional"}

function MaiKernel_Scalable(V, V_prime, params):
    T, S ← length(V), length(V_prime)
    W ← TimeAlignmentWeights(T, S, params.time_kernel)

    # ---- 1) 時間對抽樣（長序列時）
    if params.time_sampling.use:
        TS_pairs ← SampleTimePairs(W, params.time_sampling)  # 回傳一組 (t,s) index
        norm_c   ← 1.0 / Sum( W[t,s] for (t,s) in TS_pairs ) # 重要度加權修正
    else:
        TS_pairs ← AllPairs(1..T, 1..S)
        norm_c   ← 1.0

    total ← 0.0

    for (t,s) in TS_pairs:
        Zt, Zs ← V.Z[t], V_prime.Z[s]
        Mt, Ms ← V.M[t], V_prime.M[s]

        # ---- 2) 物件核近似：RFF 或 Nyström
        if params.rff.use:
            Φt ← RFF_Embed(Zt, params.rff)       # (N_t × D)
            Φs ← RFF_Embed(Zs, params.rff)       # (N'_s × D)
            K  ← Φt @ Φs^T                        # 近似 RBF 之 κ_Z
        else if params.nystrom.use:
            K  ← NystromKernel(Zt, Zs, params.nystrom, params.object_kernel)
        else:
            K  ← ObjectKernelMatrix(Zt, Zs, params.object_kernel)

        A ← ElementwiseMultiply(K, Outer(Mt, Ms))

        s_tot ← SumAll(A)
        r     ← RowSums(A)
        c     ← ColSums(A)
        fro2  ← SumAll(Square(A))

        pair_sum ← s_tot^2 - Dot(r, r) - Dot(c, c) + fro2

        total ← total + (W[t,s] * pair_sum)

    return norm_c * total


# ---- RFF 嵌入（以 RBF 為例）----
function RFF_Embed(X, rff_params):
    # X: (n×d),  取 D 個隨機特徵，γ 為 RBF 參數
    D, γ, seed ← rff_params.D, rff_params.gamma, rff_params.seed
    RandomInit(seed)
    # ω_k ~ N(0, 2γ I_d), b_k ~ U[0, 2π]
    Ω ← GaussianMatrix(d, D, mean=0, cov=2γ I)
    b ← UniformVector(D, 0, 2π)
    # φ_k(x) = sqrt(2/D) * cos(Ω^T x + b)
    return sqrt(2/D) * cos(X @ Ω + b)   # (n×D)
```

**備註**

* `SampleTimePairs` 可用 `topk`（挑最大 W 的 (t,s)）或 `proportional`（依 W 機率抽樣）；`norm_c` 確保無偏估計。
* 若物件數很多，可對每個 `(t,s)` 再做**物件子抽樣**（按行列總和比例抽）以近似 `A`，仍保可微。
* 以上所有步驟（含 soft-DTW/GAK 權重、RFF）皆為**可微**，可嵌入端到端學習。

---

## 介面建議（實作時）

* `MaiKernelBatch({V_b}, {V'_b}, params)`：回傳 Gram 矩陣 $K_{bb'}$；對大型資料，使用分塊與多工。
* 正規化：可回傳 `total / (ε + sum(W))` 或對每個 `(t,s)` 以 `pair_sum` 除以 $(\sum Mt)^2(\sum Ms)^2$ 做 scale-invariant。
* 數值穩定：`pair_sum` 理論 ≥ 0；實作中可 `max(pair_sum, 0)` 以消弭浮點誤差。

---

如果你要，我可以把這份偽程式轉成 **PyTorch/NumPy** 的參考實作（含 soft-DTW/GAK 權重計算與 RFF 模組化），以及提供一個 **SSv2 小型樣例** 的單元測試腳本。
