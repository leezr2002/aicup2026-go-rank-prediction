# 方法細節

本文對照程式說明最終提交的每個步驟。參數在 [`rank_prediction/config/final.yaml`](../rank_prediction/config/final.yaml)，程式在 `rank_prediction/src/`。段位依序編號 0–9，對應 D、C、B、A、1D、2D、3D、4D、5D、6D。

## 1. 上游分支（保存的輸入）

[`artifacts/branches/official.npz`](../rank_prediction/artifacts/branches/) 保存 800 題的上游輸出，每一項都是 10 個段位的機率：

| 欄位 | 內容 |
|---|---|
| `history` | 歷史棋力分支 |
| `opponent` | 對手分支 |
| `fingerprint` | 棋譜指紋分支 |
| `crosscolor` | 跨執色分支 |
| `kernel` | 核模型分支 |
| `policy_delta` | 策略修正，以對數機率差的形式加在 `history` 上 |
| `crosscolor_anchored` | 盤數融合用的跨執色機率 |

`training.npz` 是訓練集的同類輸出，共 22,138 筆，來自 6,321 位玩家，另含段位標籤、玩家代碼與 4 折分組。每筆都是該玩家所在折以外的模型產生的折外預測。

## 2. 輸入核對

[`src/prepare_stage.py`](../rank_prediction/src/prepare_stage.py) 的 `official_inputs` 依序檢查：

1. 三個官方 CSV 的 SHA-256 與比賽時相同。
2. 每題盤數，以及每盤棋的讓子數 `HD` 不超過 1（測試集只有 HD0 與 HD1）。
3. 題號順序與 `official.npz` 完全一致。

## 3. 初始機率與比較特徵

程式：[`src/graph.py`](../rank_prediction/src/graph.py) 的 `sample_input`。

**初始機率（anchor）**

```text
anchor = softmax( log opponent + 0.55 · clip(log fingerprint − log history, −4, 4) )
```

**比較特徵（133 維）**：對六個機率分布（anchor、history、fingerprint、crosscolor、kernel、hmm）各取：

| 特徵 | 維度 |
|---|---:|
| `√p`（10 個段位） | 10 |
| 相鄰段位的對數機率差，截斷在 ±5 後除以 4 | 9 |
| 期望段位、標準差、熵 | 3 |

其中 `hmm = softmax(log history + policy_delta)`。六個分布共 6 × 22 = 132 維，再加上盤數 `log2(max(n, 2) / 8)`，總共 133 維。

## 4. 兩人棋力差模型

程式：`graph.py` 的 `Margin`、[`src/train_stage.py`](../rank_prediction/src/train_stage.py)。

**架構**
- 共用編碼器：`133 → 48 → 24`，含 LayerNorm、SiLU、Dropout 0.12。
- 比較頭：輸入 `[a, b, a − b, a · b]`（96 維），經 `96 → 56 → 19`，含 LayerNorm、SiLU、Dropout 0.18。
- 輸出經 `2.3 · tanh(· / 2.3)` 限制範圍，最後一層以 0 初始化，訓練一開始等於不修正。
- 19 個輸出對應段位差 `d = 甲 − 乙`，從 −9 到 +9。

**訓練**
- **分折**：依保存的玩家分折做 4 折，每折各訓練一個模型，訓練與驗證的玩家不重疊。
- **抽樣**：每個 epoch 抽 36,000 對樣本。
  - 抽樣權重先依每位玩家的樣本數取反比，再依段位平衡。
  - 64% 的配對限定對方的預期段位在 ±1 級內，讓模型多學相近段位的區分。
  - 同一位玩家不和自己配對。
- **損失**：以兩人 anchor 推得的段位差先驗為基底，加上模型輸出，對真實段位差做交叉熵；另加 0.022 倍的輸出平方懲罰。
- **優化**：AdamW，學習率 0.0013、weight decay 0.045，batch 400，11 個 epoch。
- **種子**：亂數種子固定，同一環境重訓可得到相同權重。

## 5. 近鄰圖與訊息傳遞

程式：`graph.py` 的 `edges`、`factors`、`message`，以及 [`src/predict_stage.py`](../rank_prediction/src/predict_stage.py)。

Public 與 Private 各自在 400 題內建圖，兩邊不互相連接。

**選鄰居**：玩家 i、j 的距離為

```text
dist = |E_i − E_j| + 0.8 · Hellinger(anchor_i, anchor_j) + 0.13 · 特徵距離 + 0.09 · |盤數特徵差|
```

- `E` 是 anchor 的期望段位。
- 特徵距離用 18 個統計特徵（標準化後的平方距離）計算。
- 每位玩家先取 12 位最近鄰，訊息傳遞時使用前 10 位。

**邊的證據**：四個模型對 (i, j) 和 (j, i) 兩個方向的輸出取對稱平均，得到 19 種段位差的對數勢能。

**訊息傳遞**：進行兩輪。
1. 鄰居權重 `w_ij = softmax(−dist)`。
2. 對每位鄰居 j，把段位差勢能和 j 目前的機率結合，得到 i 的邊際似然。
3. 依權重取對數平均，並扣除在 anchor 下的期望值。
4. 更新 `current = softmax(log anchor + 0.48 · pooled)`。

最後的圖模型機率：

```text
graph = softmax( log anchor + 1.25 · clip(pooled, −2, 2) )
```

0.48 的折減是為了避免高度相關的鄰居證據被重複計算。

## 6. 盤數融合與決策

```text
w = interp(盤數, [5, 10, 15, 20], [0.9937, 0.9921, 0.9265, 0.6923])
p = softmax( w · log graph + (1 − w) · log crosscolor_anchored )
```

盤數越多，跨執色模型越可靠，圖模型的權重就越低。

決策依競賽計分取期望得分最高的段位：

```text
score(k) = p[k] + (p[k − 1] + p[k + 1]) / e
```

輸出 800 列 `question_id,pred_rank`，題號順序與官方範例相同。
