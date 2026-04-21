# README critical issues — all resolved

## Factually wrong
- [x] Section 1: event counts corrected (55 / 54 / 1)
- [x] Section 10: panel N corrected (380 / 54)
- [x] Section 12: LOO n corrected (54)
- [x] Section 13: 2026 date corrected to March 9
- [x] Section 7: feature count corrected to 11
- [x] Section 2: citation placeholder removed
- [x] Section 15: already-done items removed from Limitations

## Incoherence
- [x] Demand/supply framing fixed (descriptive patterns, not linear predictor)
- [x] LOO Sharpe vs negative CV R² reconciled
- [x] Secondary definition overlap framed honestly

## Missing depth
- [x] Conclusion section (Section 16) added
- [x] Section 13 expanded with sector-level 2026 actuals
- [x] Null baseline added to model tables with proper interpretation
      Key insight: null itself is ~-0.44 due to event clustering; models beating null
      are doing real work even with negative absolute CV R²
- [x] PCA scores as features — CLOSED: using outcome CARs as features is leakage;
      pre-shock sector return PCA is complex for unclear gain; not worth implementing

## Earlier TODO items (original)
- [x] XLE model
- [x] Lagged shock_type
- [x] Macro controls
- [x] Secondary shock definition
