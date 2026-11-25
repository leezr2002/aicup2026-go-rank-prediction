# 資料

## 官方競賽資料（必要）

競賽資料由 AI CUP 2026 主辦單位提供，本 repo 不重新散布。請從競賽平台取得後，放到 `data/official/rank_prediction/`：

| 檔案 | SHA-256 |
|---|---|
| `rank_prediction_test_public.csv` | `4691803a8306ccd65d93a90bd6d7ba767d57abe96c0660b262f1a2b756b84fef` |
| `rank_prediction_test_private.csv` | `d59865e24fe2f5ad742db68108a7f59c2e41138f18c4dc2eea1aa364f3e73c73` |
| `sample_submission_rank.csv` | `aa5016dbc04d7c64860e1b8727d2c2fa0b19a28a481fd3c674f82ed36f68ed16` |

程式執行前會核對上表的雜湊、題號順序、每題盤數，以及每盤棋的讓子數，任一項不符都會停止。重現提交檔與重訓都不需要官方訓練集。

## 模型產物（`rank_prediction/artifacts/`，已在 repo 內）

| 檔案 | 內容 |
|---|---|
| `branches/official.npz` | 800 題的上游分支段位機率（歷史棋力、對手、棋譜指紋、跨執色、核模型、策略修正），以及盤數融合用的跨執色機率 |
| `branches/training.npz` | 訓練集 22,138 筆折外分支機率、段位標籤、匿名玩家代碼、4 折分組 |
| `weights/fold0.pt`–`fold3.pt` | 比賽使用的四個兩人棋力差模型 |

這些檔案是模型輸出與標籤，不含任何棋譜內容。

## 外部資源

本 repo 不包含 KataGo 程式、權重或 Fox 棋譜。比賽期間使用的版本與授權列在[首頁的外部資源表](../README.md#資料與外部資源)。
