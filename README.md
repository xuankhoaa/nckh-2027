# nckh-2027

> Mamba for Vietnamese bank-stock return forecasting

This small research project reproduces the **Mamba core** from [MambaStock](https://github.com/zshicode/MambaStock) while correcting its data pipeline for the question:

> Given observations through trading day *t*, predict VCB's log return on trading day *t + 1*.

Hướng dẫn tiếng Việt từ dữ liệu đầu vào đến bảng/biểu đồ đầu ra: [`HUONG_DAN_A_Z.md`](HUONG_DAN_A_Z.md). Báo cáo kết quả nhóm được sinh tự động tại `results/BAO_CAO_KET_QUA.md` sau khi chạy toàn bộ mã.

The first experiment has downloaded and cached real daily VCB and VN-Index data from 2020-01-02 through 2026-10-09. It contains no generated market data.

## What was retained and what changed

| MambaStock component | Decision | Reason |
| --- | --- | --- |
| `external/MambaStock/mamba.py` | Retained unchanged | Selective state-space/Mamba layers are reused through `models/mamba_model.py`. |
| `external/MambaStock/pscan.py` | Retained unchanged | It provides Mamba's parallel selective scan. |
| `external/MambaStock/main.py` | Reference only | It uses each day's `pct_chg` as the label for that same row, makes one full-history sequence rather than fixed windows, has no validation split, and reports converted closing-price error rather than return error. |
| TuShare CSV files | Reference only | The project uses cached Vnstock daily OHLCV CSV files instead. |

The original `main.py` does remove `pct_chg` from its input, but still trains the output for date *t* using information from that date. That is not the required *t → t+1* task. The replacement pipeline uses only a window ending at *t* and binds its label to `return_(t+1)`.

## Files

- `get_data.py` — retrieves and caches daily OHLCV from Vnstock; it does not refetch an existing CSV unless `--force` is provided.
- `preprocess.py` — calculates `return`, `volume_change`, `high_low_range`, and `vnindex_return`; performs chronological splitting and fits `StandardScaler` only on training-period observations.
- `models/mamba_model.py` — MambaStock Mamba core with a next-day-return regression head using only the final window state.
- `models/lstm.py` — compact baseline with the identical input/output convention.
- `train.py` — runs one model, selects epoch by validation loss, evaluates the test set once, and saves outputs.
- `run_experiments.py` — runs LSTM, Mamba without VN-Index, and Mamba with VN-Index.
- `results/` — generated metrics, predictions, and plots.

## Data and API key

Create `.env` beside `get_data.py` using `.env.example` as the template:

```text
VNSTOCK_API_KEY=your_api_key_here
```

`.env` is ignored by Git. The scripts call `load_dotenv()` and rely on Vnstock's normal environment-key discovery; the key is never placed in source code or output files.

Fetch cached data:

```powershell
& C:\Users\nguye\.venv\Scripts\python.exe get_data.py
```

To extend to the other banks later:

```powershell
& C:\Users\nguye\.venv\Scripts\python.exe get_data.py --symbols VCB BID CTG MBB TCB ACB VPB STB
```

Add `--force` only when deliberately refreshing existing data.

## Leakage controls

1. `target_return = return.shift(-1)` and `target_date = time.shift(-1)`: the final input day and target day are explicitly different.
2. Each sequence has shape `[batch_size, window, num_features]`; for window 30 it ends on *t* and predicts *t+1*.
3. Split assignment uses the **target date**: train through 2023, validation in 2024, and test from 2025 onward.
4. The scaler is fitted only through the final input day belonging to training samples. Validation/test observations do not affect its mean or scale.
5. No random split is used. Training batches retain chronological ordering.

Every training run prints an alignment sample such as `input 2020-01-03 ... 2020-02-20 -> target 2020-02-21`.

## Run the complete banking-group experiment

```powershell
& C:\Users\nguye\.venv\Scripts\python.exe run_experiments.py --window 30 --epochs 75 --batch-size 128
& C:\Users\nguye\.venv\Scripts\python.exe generate_report.py --window 30
```

It writes:

- `results/metrics.csv`: `model, ticker, rmse, mae, r2, directional_accuracy` plus the window and feature-set identifiers.
- `results/predictions.csv`: target date, actual log return, and predicted log return for each model.
- Actual-vs-predicted, RMSE, MAE, directional-accuracy, and training-loss PNG charts.

## Latest complete run — real results

Run specification: 8 tickers (VCB, BID, CTG, MBB, TCB, ACB, VPB, STB), 30-day window, 2020–2023 training, 2024 validation, 2025–2026-10-12 test, seed 42, 75-epoch maximum with validation early stopping.

| Model | Mean RMSE | Mean MAE | Mean R² | Mean Directional Accuracy |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 0.026417 | 0.020189 | -1.078250 | 0.448747 |
| Mamba baseline (OHLCV-derived) | 0.053051 | 0.044755 | -7.573619 | 0.470387 |
| Improved Mamba (adds VN-Index return) | 0.024680 | 0.018623 | -0.732823 | 0.446469 |

Improved Mamba has the lowest average RMSE/MAE and highest average R² in this run; Mamba baseline has the highest average directional accuracy. This is evidence of a trade-off, not a conclusive claim that one model dominates. The complete per-ticker tables and standard deviations are generated in `results/BAO_CAO_KET_QUA.md`.
