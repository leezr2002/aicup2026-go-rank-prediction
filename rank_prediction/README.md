# rank_prediction

棋力預測的程式、設定與比賽產物。最終版本為 **domain_neighbors_isolated**：Private 0.7690426（第 10 名／175 隊），Public 0.7785244。方法說明見 [docs/method.md](../docs/method.md)。

## 各階段指令

從 repo 根目錄執行。`acceptance.py` 會依序呼叫推論與驗證；需要逐步執行時，可以直接用以下指令：

```shell
python -B rank_prediction/scripts/prepare.py
python -B rank_prediction/scripts/train.py
python -B rank_prediction/scripts/predict.py --trained
python -B rank_prediction/scripts/verify.py
```

| 指令 | 輸入 | 輸出 |
|---|---|---|
| `prepare.py` | 官方 CSV、`artifacts/branches/official.npz` | `outputs/official_inputs.npz`（檢查用，推論時會重算） |
| `train.py` | `artifacts/branches/training.npz` | `outputs/weights/fold0.pt`–`fold3.pt` |
| `predict.py` | 官方 CSV、`artifacts/branches/official.npz`、四折權重 | `outputs/submission_rank.csv` |
| `verify.py` | `outputs/submission_rank.csv`、`submission/submission_rank.csv` | `outputs/verification.json` |

- `predict.py` 預設使用比賽權重 `artifacts/weights/`；加 `--trained` 則使用 `train.py` 產生的 `outputs/weights/`。
- `config/final.yaml` 記錄圖參數（鄰居數 10、強度 1.25）、盤數權重、段位順序、模型 epoch 數，以及官方輸入與提交原檔的 SHA-256。

## 程式

| 檔案 | 內容 |
|---|---|
| `src/prepare_stage.py` | 核對官方輸入，組合初始機率與 133 維比較特徵 |
| `src/graph.py` | 比較特徵、兩人棋力差模型、近鄰圖、訊息傳遞 |
| `src/train_stage.py` | 依玩家分折訓練四個兩人棋力差模型 |
| `src/predict_stage.py` | Public、Private 分別建圖、盤數融合、段位決策、寫出 CSV |
| `src/verify_stage.py` | 呼叫根目錄的 `verify_submission.py` |

提交原檔的 SHA-256 為 `84861fd940cbde263d669308d9dcb5a11416fe622a0face181f945e9563ca814`。
