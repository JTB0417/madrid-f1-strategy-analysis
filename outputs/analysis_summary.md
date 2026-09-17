# Analysis Summary

## Decision-focused findings

- The VSC lasted 123 seconds, from lap 14 into lap 15.
- Antonelli and Verstappen entered the pits on lap 14 during the VSC; Norris stopped on lap 15 after green-flag running resumed.
- Norris's recorded pit-lane transit was 3.3 seconds longer than Antonelli's (35.1s vs 31.8s).
- Position data shows Norris falling from the lead group to fifth during the pit sequence before recovering to third.
- The winner finished 4.351 seconds ahead of second and 5.089 seconds ahead of third.
- All three podium contenders showed improving net hard-stint lap-time trends. Fuel burn therefore outweighed observed tire degradation in the raw trend; this must not be interpreted as pure tire wear.
- Random Forest produced the lowest held-out-driver MAE: 1.397 seconds across 1,024 cleaned laps.

## Engineering recommendation

At a circuit where passing was difficult, track position had unusually high value. Once the VSC was deployed near the planned pit window, the preferred decision was to pit immediately when operationally possible. A strategy tool should combine a live VSC state, estimated pit-lane loss, traffic on pit exit, and stop-time uncertainty rather than evaluating tire pace alone.

## Limitations

This is a single-race observational analysis. Fuel load, traffic, tire condition, driver pace, and setup are not independently measured. The models identify useful patterns but cannot prove a causal tire-degradation rate or reconstruct an exact alternate race result.
