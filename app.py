"""AQUAGUARD-X software dashboard (no board or UART required)."""

from io import BytesIO
from json import dumps
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import altair as alt
import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).with_name("AQUAGUARD_X_COMPLETE_DATASET_1S_V2.csv")
C_RESULTS = Path(__file__).with_name("c_results.csv")
C_VALVE_RESULTS = Path(__file__).with_name("c_valve_results.csv")
REQUIRED = {
    "scenario", "timestamp", "sensor_main_flow", "sensor_main_pressure",
    "sensor_kitchen_flow", "sensor_bathroom_flow", "tank_level",
}
EXPECTED = {
    "normal": "NORMAL", "kitchen_usage": "LEGITIMATE_USAGE",
    "bathroom_usage": "LEGITIMATE_USAGE", "simultaneous_usage": "LEGITIMATE_USAGE",
    "small_leak": "SMALL_LEAK", "large_leak": "LARGE_LEAK",
    "burst": "BURST", "blockage": "BLOCKAGE",
    "sensor_fault": "SENSOR_FAULT", "sensor_bias": "SENSOR_FAULT",
    "sensor_drift": "SENSOR_FAULT", "sensor_spikes": "SENSOR_FAULT",
    "sensor_disconnected": "SENSOR_FAULT", "valve_fault": "SMALL_LEAK",
}


# Expected controller outcomes for this test plan. Leak isolation assumes
# the leaking branch responds to the close command, except in valve_fault.
NO_VERIFICATION = "NOT REQUIRED"
EXPECTED_VERIFICATION = dict.fromkeys(EXPECTED, NO_VERIFICATION)
EXPECTED_VERIFICATION.update({
    "small_leak": "SUCCESS", "large_leak": "SUCCESS", "burst": "SUCCESS",
    "valve_fault": "FAILED",
})
SOLUTIONS = {
    "large_leak": "Rerun with WNTR pressure-driven demand and apply the commanded bathroom valve closure to the model. Feed the resulting branch readings back to the C module, then check that bathroom flow drops and kitchen flow remains available.",
    "burst": "Replay the burst with a simulated valve closure and feed the post-close readings back to the C module. Verify that the isolated branch loses flow before reporting success. Escalate if flow stays high.",
    "blockage": "Use bathroom branch pressure and a sustained flow restriction to classify BLOCKAGE before leak isolation. Replay with the branch pressure included and check that no leak close command is issued.",
    "sensor_fault": "Pass a bathroom flow sensor health flag into the C module and check sensor validity before applying leak rules. Classify invalid readings as SENSOR_FAULT and suppress leak isolation based on those readings.",
    "sensor_bias": "Calibrate the bathroom flow sensor at zero flow and use a sustained offset check against the main and kitchen readings. Flag the offset as SENSOR_FAULT before leak logic runs.",
    "sensor_drift": "Track the bathroom sensor baseline during known no-flow periods. Flag a persistent baseline shift as SENSOR_FAULT before treating the readings as usage.",
    "sensor_spikes": "Use a median filter or rate-of-change check to reject isolated outliers. Flag repeated spikes as SENSOR_FAULT and confirm with a replay that brief spikes do not trigger isolation.",
    "sensor_disconnected": "Include a sensor health flag or signal timeout in the packet so the C module can distinguish a missing signal from a valid zero-flow reading. Check for SENSOR_FAULT and no false burst alarm on replay.",
}


# Operational guidance for a person responding to the corresponding real fault.
# A replay match can still represent a fault that needs maintenance.
HUMAN_RESPONSE = {
    "small_leak": ("REPAIR REQUIRED", "Plumber or building maintenance", "Inspect and repair the isolated pipe, then check for leaks before restoring water."),
    "large_leak": ("URGENT", "Building maintenance or plumber", "Check whether isolation succeeded. If water is still escaping, have authorized maintenance isolate the affected supply manually and repair the pipe."),
    "burst": ("URGENT", "Building emergency maintenance or plumber", "Notify building maintenance immediately. Have authorized staff isolate the affected supply and arrange pipe repair before restoring water."),
    "blockage": ("INSPECTION REQUIRED", "Plumber or building maintenance", "Inspect the restricted branch and clear or repair the obstruction. Check that branch pressure and flow return to normal."),
    "sensor_fault": ("INSPECTION REQUIRED", "Instrumentation technician", "Confirm flow independently, then inspect the bathroom flow sensor and its wiring. Repair or replace the sensor and validate its readings."),
    "sensor_bias": ("INSPECTION REQUIRED", "Instrumentation technician", "Check the sensor against a known zero-flow reading and a reference flow. Recalibrate it or replace it if the offset persists."),
    "sensor_drift": ("INSPECTION REQUIRED", "Instrumentation technician", "Compare the bathroom sensor with a reference over time. Recalibrate or replace it and confirm readings remain stable."),
    "sensor_spikes": ("INSPECTION REQUIRED", "Instrumentation technician", "Check the sensor cable, connectors and power for intermittent faults. Repair the cause and verify stable readings during steady flow."),
    "sensor_disconnected": ("INSPECTION REQUIRED", "Instrumentation technician", "Check the connector, cable and power. Confirm that the sensor reports valid zero-flow and flowing readings after repair."),
    "valve_fault": ("URGENT IF LEAK CONTINUES", "Building maintenance or valve technician", "Check whether water is still escaping and have authorized staff isolate the supply manually if needed. Inspect the valve, actuator and wiring, then retest closure and confirm that branch flow drops."),
}


def human_response(scenario: str, expected: str, rows: pd.DataFrame, verifications: list[str], commands: list[str]) -> dict:
    response = HUMAN_RESPONSE.get(scenario)
    if response is None and (rows.empty or expected == "UNDEFINED"):
        response = ("REVIEW REQUIRED", "Dashboard operator",
                    "Confirm the scenario definition and obtain the missing replay evidence before deciding what maintenance is needed.")
    if response is None and (
        any(value in {"FAILED", "UNCERTAIN"} for value in verifications)
        or (commands and not verifications)
    ):
        response = ("REVIEW REQUIRED", "Building maintenance or dashboard operator",
                    "Check the affected branch and confirm valve position and flow before relying on automatic isolation.")
    status, contact, action = response or ("NONE EXPECTED", "", "")
    return {"status": status, "contact": contact, "action": action}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(sorted(missing))}")
    return frame.sort_values(["scenario", "timestamp"]).reset_index(drop=True)


def make_sequence(source: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align the actual C module replay with the V2 sensor readings."""
    rows = source.loc[source.scenario.eq(scenario)].copy().reset_index(drop=True)
    rows.insert(0, "packet", range(1, len(rows) + 1))
    result_file = C_VALVE_RESULTS if scenario in {"small_leak", "valve_fault"} else C_RESULTS
    results = load_results(str(result_file))
    c = results.loc[results.scenario.eq(scenario)].copy()
    if c.empty:
        raise ValueError(f"No C results for {scenario}")
    rows = rows.merge(c, left_on=["scenario", "packet"], right_on=["scenario", "packet_number"], validate="one_to_one")
    rows = rows.rename(columns={"command": "c_command", "state": "c_state"})
    rows["valve_state"] = "OPEN"
    if scenario in {"small_leak", "valve_fault"}:
        after = rows.phase.ne("DATASET")
        if scenario == "small_leak":
            rows.loc[after, "sensor_main_flow"] = rows.loc[after, "sensor_kitchen_flow"].to_numpy()
            rows.loc[after, "sensor_bathroom_flow"] = 0.0
            rows.loc[after, "valve_state"] = "CLOSED"
        else:
            rows.loc[after, "valve_state"] = "STUCK OPEN"
    rows.loc[rows.c_command.eq("VALVE,BATHROOM,CLOSE"), "valve_state"] = "CLOSE COMMANDED"
    events = []
    previous_diagnosis = None
    for _, row in rows.iterrows():
        if row.diagnosis != previous_diagnosis:
            events.append((int(row.packet), "C DIAGNOSIS", row.diagnosis))
            previous_diagnosis = row.diagnosis
        if row.c_command:
            events.append((int(row.packet), "C COMMAND", row.c_command))
        if row.verification:
            events.append((int(row.packet), "C VERIFICATION", row.verification))
    log = pd.DataFrame(events, columns=["packet", "event", "detail"])
    return rows, log


def valve_flow_comparison(sequence: pd.DataFrame):
    """Pair the first close command with its first subsequent verdict."""
    close_rows = sequence.loc[sequence.c_command.eq("VALVE,BATHROOM,CLOSE")]
    if close_rows.empty:
        return None
    close = close_rows.iloc[0]
    verdict_rows = sequence.loc[sequence.packet.gt(close.packet) & sequence.verification.ne("")]
    if verdict_rows.empty:
        return None
    verdict = verdict_rows.iloc[0]
    if pd.isna(close.sensor_bathroom_flow) or pd.isna(verdict.sensor_bathroom_flow):
        return None
    return close, verdict


@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"scenario", "packet_number", "diagnosis", "state", "command", "verification", "phase"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing C result columns: {', '.join(sorted(missing))}")
    return frame


def assess(scenario: str, rows: pd.DataFrame) -> dict:
    expected = EXPECTED.get(scenario, "UNDEFINED")
    expected_verification = EXPECTED_VERIFICATION.get(scenario, "UNDEFINED")
    diagnoses = list(dict.fromkeys(rows.diagnosis))
    verifications = [value for value in rows.verification if value]
    commands = [value for value in rows.c_command if value]
    false_alarm = expected in {"NORMAL", "LEGITIMATE_USAGE", "SENSOR_FAULT", "BLOCKAGE"} and any(
        diagnosis in {"SMALL_LEAK", "LARGE_LEAK", "BURST"} for diagnosis in diagnoses
    )
    diagnosis_matches = expected in diagnoses and not false_alarm
    if rows.empty:
        recorded = "NO REPLAY DATA"
        verification_matches = False
    elif verifications:
        recorded = ", ".join(
            f"{row.verification} (packet {row.packet})"
            for row in rows.itertuples() if row.verification
        )
        verification_matches = all(value == expected_verification for value in verifications)
    elif commands:
        recorded = "COMMAND ISSUED · NO VERDICT"
        verification_matches = False
    else:
        recorded = "NO VERIFICATION RECORDED"
        verification_matches = expected_verification == NO_VERIFICATION

    matches = diagnosis_matches and verification_matches
    solution = ""
    if not matches:
        if rows.empty:
            solution = "Export the missing C replay results for this scenario and reload the dashboard."
        elif expected == "UNDEFINED":
            solution = "Define the expected diagnosis and valve verification outcome in the test plan."
        else:
            solution = SOLUTIONS.get(scenario, (
                "Review the first divergent packet and check the C diagnosis and valve response rules. "
                "Rerun with the expected valve behavior and capture the command and verdict."
            ))
    return {
        "scenario": scenario, "expected_diagnosis": expected,
        "recorded_diagnoses": diagnoses, "commands": commands,
        "expected_verification": expected_verification,
        "recorded_verification": recorded, "verification": verifications,
        "diagnosis_matches": diagnosis_matches, "verification_matches": verification_matches,
        "result": "MATCH" if matches else "MISMATCH", "suggested_solution": solution,
        "human_intervention": human_response(scenario, expected, rows, verifications, commands),
    }


def overview(names: list[str]) -> pd.DataFrame:
    entries = []
    for name in names:
        file = C_VALVE_RESULTS if name in {"small_leak", "valve_fault"} else C_RESULTS
        results = load_results(str(file))
        rows = results.loc[results.scenario.eq(name)].rename(
            columns={"packet_number": "packet", "command": "c_command"}
        )
        outcome = assess(name, rows)
        entries.append({
            "Scenario": name.replace("_", " ").title(),
            "Expected verification": outcome["expected_verification"],
            "Recorded verification": outcome["recorded_verification"],
            "Verification check": "MATCH" if outcome["verification_matches"] else "MISMATCH",
            "Expected diagnosis": outcome["expected_diagnosis"],
            "Recorded C diagnoses": ", ".join(outcome["recorded_diagnoses"]),
            "Diagnosis check": "MATCH" if outcome["diagnosis_matches"] else "MISMATCH",
            "Result": outcome["result"],
            "Human intervention": outcome["human_intervention"]["status"],
            "Who to contact": outcome["human_intervention"]["contact"],
            "Recommended action": outcome["human_intervention"]["action"],
            "Suggested solution": outcome["suggested_solution"],
        })
    return pd.DataFrame(entries)


def report_zip(scenario: str, sequence: pd.DataFrame, events: pd.DataFrame) -> bytes:
    outcome = assess(scenario, sequence)
    outcome["source"] = "Recorded laptop C module replay with simulated valve feedback for small_leak and valve_fault"
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("readings.csv", sequence.to_csv(index=False))
        archive.writestr("events.csv", events.to_csv(index=False))
        archive.writestr("outcome.json", dumps(outcome, indent=2))
    return buffer.getvalue()


def marked_chart(frame: pd.DataFrame, columns: list[str], unit: str) -> alt.Chart:
    long = frame.melt(id_vars="packet", value_vars=columns, var_name="Sensor", value_name=unit)
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("packet:Q", title="Packet"),
        y=alt.Y(f"{unit}:Q", title=unit), color="Sensor:N",
    )
    events = frame.loc[frame.c_command.eq("VALVE,BATHROOM,CLOSE") | frame.verification.ne("")].copy()
    if events.empty:
        return lines.properties(height=260).interactive()
    events["Marker"] = events.apply(
        lambda row: "Close command" if row.c_command == "VALVE,BATHROOM,CLOSE"
        else f"Verification: {row.verification}", axis=1,
    )
    rules = alt.Chart(events).mark_rule(strokeDash=[5, 4]).encode(
        x="packet:Q", color=alt.Color("Marker:N", scale=alt.Scale(
            domain=["Close command", "Verification: SUCCESS", "Verification: FAILED", "Verification: UNCERTAIN"],
            range=["#f59e0b", "#16a34a", "#dc2626", "#9333ea"],
        )), tooltip=["packet:Q", "Marker:N"],
    )
    return (lines + rules).properties(height=260).interactive()


st.set_page_config(page_title="AQUAGUARD-X", layout="wide", page_icon="💧")
st.title("AQUAGUARD-X")
st.caption("V2 WNTR dataset · recorded laptop C module results · 1-second packets")

try:
    data = load_data(str(DATA_PATH))
except (OSError, ValueError) as exc:
    st.error(f"Could not load the bundled CSV: {exc}")
    st.stop()

scenario_order = [
    "normal", "kitchen_usage", "bathroom_usage", "simultaneous_usage",
    "small_leak", "large_leak", "burst", "blockage", "sensor_fault",
    "sensor_bias", "sensor_drift", "sensor_spikes", "sensor_disconnected",
    "valve_fault",
]
available = set(data.scenario.unique())
options = [name for name in scenario_order if name in available]
options += sorted(available - set(options))
scenario_number = {name: number for number, name in enumerate(options, 1)}
scenario = st.selectbox(
    "Scenario",
    options,
    format_func=lambda name: f"{scenario_number[name]:02d} / {len(options):02d}  ·  {name.replace('_', ' ').title()}",
)
if st.button("Run demo", type="primary"):
    st.session_state["selected_demo"] = scenario
selected = st.session_state.get("selected_demo", scenario)
if selected != scenario:
    st.info(f"Showing last run: {selected}. Click Run demo to run {scenario}.")
st.caption(f"Showing scenario {scenario_number[selected]} of {len(options)}: {selected.replace('_', ' ').title()}")

st.subheader("Scenario overview")
st.caption("Select a scenario for the full diagnosis, suggested fix, and maintenance guidance.")
summary = overview(options)[[
    "Scenario", "Expected verification", "Recorded verification",
    "Verification check", "Result", "Human intervention",
]]
st.dataframe(summary, use_container_width=True, hide_index=True,
             height=min(560, 36 * (len(options) + 1)))


@st.fragment(run_every="1s")
def show_replay(name: str) -> None:
    sequence, full_log = make_sequence(data, name)
    outcome = assess(name, sequence)
    st.subheader("Scenario verification")
    expected, recorded = st.columns(2)
    expected.metric("Expected verification", outcome["expected_verification"])
    recorded.write("**Recorded verification**")
    recorded.write(outcome["recorded_verification"])
    st.write(f"**Expected diagnosis:** {outcome['expected_diagnosis']}")
    st.write(f"**Recorded diagnoses:** {', '.join(outcome['recorded_diagnoses'])}")
    diagnosis_check = "MATCH" if outcome["diagnosis_matches"] else "MISMATCH"
    verification_check = "MATCH" if outcome["verification_matches"] else "MISMATCH"
    st.write(f"Diagnosis check · **{diagnosis_check}**    |    Verification check · **{verification_check}**")
    if outcome["suggested_solution"]:
        st.warning(f"Suggested solution\n\n{outcome['suggested_solution']}")

    with st.container(border=True):
        st.subheader("Action taken & next step")
        commands = [
            f"{row.c_command.replace(',', ' ').title()} (packet {row.packet})"
            for row in sequence.itertuples() if row.c_command
        ]
        st.write("**Valve actions** · " + (" → ".join(commands) or "No valve command recorded"))
        st.write(f"**Valve verification** · {outcome['recorded_verification']}")
        response = outcome["human_intervention"]
        if response["action"]:
            st.write(f"**Next step on site** · {response['status']} · {response['contact']}")
            st.write(response["action"])
        else:
            st.write("**Next step on site** · Continue monitoring")

        if name in {"small_leak", "valve_fault"}:
            comparison = valve_flow_comparison(sequence)
            if comparison is not None:
                close, verdict = comparison
                st.write("**Bathroom flow during simulated valve test**")
                before, after = st.columns(2)
                before.metric(f"At close command · packet {close.packet}", f"{close.sensor_bathroom_flow:.2f} L/min")
                after.metric(f"At verification · packet {verdict.packet}", f"{verdict.sensor_bathroom_flow:.2f} L/min")

    position_key = f"packet_{name}"
    playing_key = f"playing_{name}"
    if position_key not in st.session_state:
        st.session_state[position_key] = 1
    if playing_key not in st.session_state:
        st.session_state[playing_key] = False

    st.subheader("Packet playback")
    play, pause, reset = st.columns(3)
    if play.button("▶ Play", key=f"play_{name}"):
        if st.session_state[position_key] >= len(sequence):
            st.session_state[position_key] = 1
        st.session_state[playing_key] = True
    if pause.button("⏸ Pause", key=f"pause_{name}"):
        st.session_state[playing_key] = False
    if reset.button("↺ Reset", key=f"reset_{name}"):
        st.session_state[playing_key] = False
        st.session_state[position_key] = 1
    if st.session_state[playing_key]:
        st.session_state[position_key] = min(st.session_state[position_key] + 1, len(sequence))
        if st.session_state[position_key] == len(sequence):
            st.session_state[playing_key] = False

    packet = st.slider("Inspect packet", 1, len(sequence), key=position_key)
    visible = sequence.iloc[:packet].copy()
    current = visible.iloc[-1]
    log = full_log.loc[full_log.packet.le(packet)]
    if name in {"small_leak", "valve_fault"}:
        st.caption("After the C close command, a WNTR-derived valve response was simulated. The C verification result is shown at packet 53.")
    else:
        st.caption("The replay uses unchanged valve-open CSV readings. A FAILED verification here does not test a physical valve.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Main flow", f"{current.sensor_main_flow:.2f} L/min")
    c2.metric("Main pressure", f"{current.sensor_main_pressure:.2f} bar")
    c3.metric("Bathroom valve", current.valve_state)
    verdicts = visible.loc[visible.verification.ne(""), "verification"]
    c4.metric("Verification through this packet", verdicts.iloc[-1] if not verdicts.empty else (
        "PENDING" if visible.c_command.eq("VALVE,BATHROOM,CLOSE").any() else "NOT STARTED"
    ))
    st.metric("C diagnosis", current.diagnosis)
    c5, c6, c7 = st.columns(3)
    c5.metric("Kitchen flow", f"{current.sensor_kitchen_flow:.2f} L/min")
    c6.metric("Bathroom flow", f"{current.sensor_bathroom_flow:.2f} L/min")
    c7.metric("Tank level", f"{current.tank_level:.2f}%")

    st.subheader("Bathroom sensor health")
    # The bundled five-field packet has no sensor-status bit or bathroom pressure.
    if name == "sensor_disconnected":
        st.error("Dataset label: disconnected bathroom flow sensor. Its reported flow is zero during bathroom usage; the C result above is shown separately.")
    elif pd.isna(current.sensor_bathroom_flow) or current.sensor_bathroom_flow < 0:
        st.warning("Invalid bathroom flow reading in this packet.")
    elif name in {"sensor_fault", "sensor_bias", "sensor_drift", "sensor_spikes"}:
        st.warning("Dataset label: simulated bathroom sensor anomaly. The five-field replay cannot independently confirm sensor health.")
    else:
        st.success("Bathroom flow reading is present and nonnegative. Independent sensor health is unavailable in this dataset.")

    st.subheader("Flow over packets")
    st.altair_chart(marked_chart(visible, ["sensor_main_flow", "sensor_kitchen_flow", "sensor_bathroom_flow"], "L/min"), use_container_width=True)
    st.subheader("Main pressure over packets")
    st.altair_chart(marked_chart(visible, ["sensor_main_pressure"], "bar"), use_container_width=True)
    st.caption("Dashed lines mark the recorded close command and verification packet, when present.")

    st.subheader("Event log")
    st.dataframe(log, use_container_width=True, hide_index=True)
    st.download_button(
        "Download scenario report ZIP", report_zip(name, sequence, full_log),
        file_name=f"aquaguard_{name}_report.zip", mime="application/zip",
        key=f"download_{name}",
    )
    with st.expander("Packet data"):
        st.dataframe(visible[[
            "packet", "timestamp", "phase", "sensor_main_flow",
            "sensor_main_pressure", "sensor_kitchen_flow", "sensor_bathroom_flow",
            "valve_state", "diagnosis", "verification", "c_state", "c_command",
        ]], use_container_width=True, hide_index=True)


show_replay(selected)
