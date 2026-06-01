# grid-anomaly-detector
A live system that continuously ingests electricity frequency data from ENTSO-E (European grid operator), detects anomalies as they happen, handles concept drift over time, and displays results on a running dashboard. It stays alive 24/7 via GitHub Actions.

Built against the backdrop of Germany's accelerating energy transition, where negative electricity prices hit -€200/MWh in April 2026 due to renewable surges overwhelming demand, and grid operators face over 500 hours of price volatility annually. Real-time anomaly detection on ENTSO-E data directly addresses the operational monitoring challenges created by this instability.

## Baseline Failure Analysis

Rolling Z-score (window=72, threshold=2.5) on 13,445 readings:

- Overall flag rate: 8.88% (target: <3%)
- False positive rate on known stable week: 11.17%
- Monday flags 4x more than Sunday despite no real events
  (weekly seasonality blindness)
- Anomalies cluster at hours 0, 5-7, 21-23
  (daily transition points flagged as anomalies)
- 137 consecutive streaks longer than 3 readings
  (drift instability causing sustained false alarms)
- Longest false alarm streak: 12 consecutive readings

Conclusion: Z-score has no awareness of time-of-day or 
day-of-week patterns. Needs a model that understands 
temporal context. Moving to LSTM residual detection.