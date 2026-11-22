# AQUAGUARD-X C results dashboard

The bundled V2 WNTR CSV contains 854 one-second records (14 scenarios).
`c_results.csv` records the output of Person 1's unchanged portable C modules
running in a laptop adapter. `c_valve_results.csv` additionally supplies a
WNTR-derived working or stuck bathroom-valve response after the C code issues
its close command. Those two demonstrations end at the first C verification.

No C executable runs on Windows when using this dashboard: its outputs were
recorded ahead of time because Windows Application Control blocks the newly
compiled executable on the target laptop. This is a replay of actual laptop C
module outputs, not STM32 firmware execution, live WNTR, UART, or hardware.

Start PowerShell in this folder:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Select a scenario and click **Run demo**. For `small_leak`, packet 47 has the
close command and packet 53 has `SUCCESS`; for `valve_fault`, packet 53 has
`FAILED`. The other scenarios show what the unchanged C code actually
classified. It does not reliably diagnose blockage and some sensor faults.

The scenario overview compares expected and recorded diagnoses and valve
verification for all 14 scenarios. The selected scenario shows the full-run
comparison and a suggested solution for any mismatch. Leak and burst scenarios
assume successful isolation with a working valve. The stuck-valve scenario
expects FAILED. Normal operation, legitimate usage, blockage and sensor faults
expect no valve verification. Large leak and burst use valve-open replay data,
so their recorded FAILED verdicts call for a rerun with simulated valve feedback.
The dashboard also shows when a real incident needs human intervention, who
should respond, and what they should check or repair. A stuck-valve test can
match its expected FAILED verdict while still calling for maintenance. A
successful leak isolation still calls for pipe repair. The downloadable outcome
summary includes the checks, suggested solution and human-intervention guidance.
Use Play, Pause, Reset, or the packet slider to inspect one-second readings.
Dashed graph markers show recorded close and verification events. The sensor
health panel identifies dataset-labeled faults without claiming that the C code
detected them. Download a scenario report ZIP for the full readings, event log,
and outcome summary.
