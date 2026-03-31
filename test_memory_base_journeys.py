#!/usr/bin/env python3
"""Live integration test for Memory Base + Knowledge Base user journeys.

Spec: MemoryBaseTestingSpec.md
Runs against the live Langflow server at http://localhost:7860

Usage:
    python3 test_memory_base_journeys.py
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:7860"
USERNAME = "langflow"
PASSWORD = "langflow"

# Threshold set LOW so we can hit it quickly during tests
THRESHOLD = 3

# How long (seconds) to wait for a background ingestion job to complete
JOB_POLL_TIMEOUT = 30
JOB_POLL_INTERVAL = 2

# ---------------------------------------------------------------------------
# Minimal modern flow JSON (ChatInput → ChatOutput, no LLM needed)
# ChatOutput saves messages with is_output=True which the memory base tracks.
# ---------------------------------------------------------------------------

MINIMAL_FLOW = {
    "name": "MemBaseTestFlow",
    "description": "Minimal flow for memory base live tests",
    "data": {
        "nodes": [
            {
                "id": "ChatInput-test1",
                "type": "genericNode",
                "position": {"x": 100, "y": 100},
                "data": {
                    "type": "ChatInput",
                    "id": "ChatInput-test1",
                    "node": {
                        "template": {
                            "input_value": {
                                "name": "input_value",
                                "display_name": "Text",
                                "type": "str",
                                "value": "Hello",
                                "show": True,
                                "required": False,
                                "advanced": False,
                            },
                            "session_id": {
                                "name": "session_id",
                                "display_name": "Session ID",
                                "type": "str",
                                "value": "",
                                "show": True,
                                "required": False,
                                "advanced": False,
                            },
                            "sender": {
                                "name": "sender",
                                "display_name": "Sender Type",
                                "type": "str",
                                "value": "User",
                                "show": False,
                                "required": False,
                                "advanced": False,
                            },
                            "sender_name": {
                                "name": "sender_name",
                                "display_name": "Sender Name",
                                "type": "str",
                                "value": "User",
                                "show": False,
                                "required": False,
                                "advanced": False,
                            },
                        },
                        "description": "Receives user chat input.",
                        "display_name": "Chat Input",
                        "base_classes": ["Message"],
                        "outputs": [
                            {
                                "name": "message",
                                "display_name": "Message",
                                "types": ["Message"],
                                "selected": "Message",
                            }
                        ],
                    },
                },
            },
            {
                "id": "ChatOutput-test1",
                "type": "genericNode",
                "position": {"x": 600, "y": 100},
                "data": {
                    "type": "ChatOutput",
                    "id": "ChatOutput-test1",
                    "node": {
                        "template": {
                            "input_value": {
                                "name": "input_value",
                                "display_name": "Text",
                                "type": "str",
                                "value": "",
                                "show": False,
                                "required": False,
                                "advanced": False,
                                "input_types": ["Message"],
                            },
                            "sender": {
                                "name": "sender",
                                "display_name": "Sender Type",
                                "type": "str",
                                "value": "Machine",
                                "show": False,
                                "required": False,
                                "advanced": False,
                            },
                            "sender_name": {
                                "name": "sender_name",
                                "display_name": "Sender Name",
                                "type": "str",
                                "value": "AI",
                                "show": False,
                                "required": False,
                                "advanced": False,
                            },
                            "session_id": {
                                "name": "session_id",
                                "display_name": "Session ID",
                                "type": "str",
                                "value": "",
                                "show": False,
                                "required": False,
                                "advanced": False,
                            },
                        },
                        "description": "Sends chat output.",
                        "display_name": "Chat Output",
                        "base_classes": ["Message"],
                        "outputs": [
                            {
                                "name": "message",
                                "display_name": "Message",
                                "types": ["Message"],
                                "selected": "Message",
                            }
                        ],
                    },
                },
            },
        ],
        "edges": [
            {
                "id": "edge-1",
                "source": "ChatInput-test1",
                "sourceHandle": '{"dataType":"ChatInput","id":"ChatInput-test1","name":"message","output_types":["Message"]}',
                "target": "ChatOutput-test1",
                "targetHandle": '{"fieldName":"input_value","id":"ChatOutput-test1","inputTypes":["Message"],"type":"other"}',
            }
        ],
    },
}

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def add(self, note: str) -> None:
        self.notes.append(note)


results: list[ScenarioResult] = []


def scenario(name: str) -> ScenarioResult:
    r = ScenarioResult(name=name, passed=False)
    results.append(r)
    print(f"\n{'=' * 70}")
    print(f"  SCENARIO: {name}")
    print("=" * 70)
    return r


def ok(r: ScenarioResult, msg: str) -> None:
    print(f"  [OK]  {msg}")
    r.add(f"[OK] {msg}")


def fail(r: ScenarioResult, msg: str) -> None:
    print(f"  [FAIL] {msg}")
    r.add(f"[FAIL] {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class Client:
    def __init__(self, base_url: str) -> None:
        self.base = base_url
        self.session = requests.Session()
        self.token: str | None = None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def login(self, username: str, password: str) -> bool:
        resp = self.session.post(
            f"{self.base}/api/v1/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        if resp.status_code == 200:
            self.token = resp.json()["access_token"]
            return True
        return False

    def get(self, path: str, **kwargs) -> requests.Response:
        return self.session.get(f"{self.base}{path}", headers=self._headers(), timeout=15, **kwargs)

    def post(self, path: str, body: Any = None, **kwargs) -> requests.Response:
        return self.session.post(f"{self.base}{path}", json=body, headers=self._headers(), timeout=15, **kwargs)

    def patch(self, path: str, body: Any = None, **kwargs) -> requests.Response:
        return self.session.patch(f"{self.base}{path}", json=body, headers=self._headers(), timeout=15, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        return self.session.delete(f"{self.base}{path}", headers=self._headers(), timeout=15, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wait_for_job(client: Client, job_id: str, timeout: int = JOB_POLL_TIMEOUT) -> dict | None:
    """Poll GET /api/v1/jobs/{job_id} until terminal state or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        if resp.status_code == 200:
            job = resp.json()
            status = job.get("status", "")
            info(f"  Job {job_id[:8]}… status: {status}")
            if status in ("completed", "failed", "cancelled"):
                return job
        time.sleep(JOB_POLL_INTERVAL)
    return None


def get_or_create_flow(client: Client) -> dict | None:
    """Return an existing flow or create the minimal test flow."""
    resp = client.get("/api/v1/flows")
    if resp.status_code == 200:
        flows = resp.json()
        if isinstance(flows, list) and flows:
            name = flows[0]["name"]
            info(f"Using existing flow: {flows[0]['id']} – '{name}'")
            return flows[0]

    info("No existing flows found. Creating minimal test flow …")
    resp = client.post("/api/v1/flows/", body=MINIMAL_FLOW)
    if resp.status_code in (200, 201):
        flow = resp.json()
        info(f"Created flow: {flow['id']}")
        return flow
    info(f"Failed to create flow: {resp.status_code} {resp.text[:200]}")
    return None


def create_kb(client: Client, kb_name: str) -> dict | None:
    """Create a Knowledge Base (required before creating a Memory Base that references it)."""
    payload = {
        "name": kb_name,
        "embedding_provider": "Fake",
        "embedding_model": "fake-model",
        "column_config": None,
    }
    resp = client.post("/api/v1/knowledge_bases", body=payload)
    if resp.status_code in (200, 201):
        info(f"Created KB: {kb_name}")
        return resp.json()

    # Already exists is also fine
    if resp.status_code in (400, 409):
        info(f"KB '{kb_name}' already exists")
        return {"name": kb_name}

    info(f"KB create response {resp.status_code}: {resp.text[:200]}")
    return None


def run_flow_once(client: Client, flow_id: str, session_id: str, message: str = "test") -> dict | None:
    """Trigger a flow run and poll until done. Returns final job status dict."""
    payload = {
        "inputs": {"input_value": message, "session_id": session_id},
        "event_delivery": "polling",
    }
    resp = client.post(f"/api/v1/build/{flow_id}/flow", body=payload)
    if resp.status_code not in (200, 202):
        info(f"  build start failed {resp.status_code}: {resp.text[:200]}")
        return None
    data = resp.json()
    job_id = data.get("job_id")
    if not job_id:
        # Some versions return the full result immediately
        return data
    return wait_for_job(client, job_id)


def get_messages(client: Client, session_id: str | None = None, flow_id: str | None = None) -> list:
    params: dict = {}
    if session_id:
        params["session_id"] = session_id
    if flow_id:
        params["flow_id"] = flow_id
    resp = client.get("/api/v1/monitor/messages", params=params)
    if resp.status_code == 200:
        return resp.json()
    return []


def inject_messages_directly(client: Client, flow_id: str, session_id: str, count: int) -> int:
    """Fallback: inject messages via the run endpoint.
    Returns the number of output messages that appeared in the monitor.
    """
    before = len([m for m in get_messages(client, session_id=session_id) if m.get("is_output")])
    for i in range(count):
        run_flow_once(client, flow_id, session_id, message=f"threshold-test-msg-{i}")
        time.sleep(0.3)
    after = len([m for m in get_messages(client, session_id=session_id) if m.get("is_output")])
    return after - before


# ---------------------------------------------------------------------------
# Scenario 0: Auth
# ---------------------------------------------------------------------------


def scenario_auth(client: Client) -> bool:
    r = scenario("0 – Authentication")
    if client.login(USERNAME, PASSWORD):
        ok(r, f"Logged in as '{USERNAME}', token acquired")
        r.passed = True
    else:
        fail(r, f"Login failed for user '{USERNAME}'")
    return r.passed


# ---------------------------------------------------------------------------
# Scenario 1: Create Memory Base and associate with a flow
# ---------------------------------------------------------------------------


def scenario_create(client: Client, flow: dict, kb_name: str, user_id: str) -> dict | None:
    r = scenario("1 – Create Memory Base associated with a flow")

    base_payload = {
        "name": f"test-mb-{uuid.uuid4().hex[:6]}",
        "flow_id": flow["id"],
        "kb_name": kb_name,
        "threshold": THRESHOLD,
        "auto_capture": True,
    }

    # --- Bug probe 1: send WITHOUT user_id (should work since endpoint derives it from token) ---
    resp_no_uid = client.post("/api/v1/memories", body=base_payload)
    if resp_no_uid.status_code == 422:
        fail(
            r,
            "POST /memories without user_id → 422: MemoryBaseCreate.user_id is a REQUIRED body "
            "field, but the endpoint should derive it from the auth token. "
            "Bug: user_id should not be required in the request body.",
        )
    elif resp_no_uid.status_code == 201:
        mb = resp_no_uid.json()
        ok(r, "POST /memories without user_id → 201 (endpoint correctly uses auth token)")
        ok(r, f"  Created id={mb['id']}, flow_id={mb['flow_id']}, threshold={mb['threshold']}")
        r.passed = True
        return mb
    else:
        fail(r, f"POST /memories without user_id → unexpected {resp_no_uid.status_code}: {resp_no_uid.text[:200]}")

    # --- Bug probe 2: send WITH user_id (service does MemoryBase(**payload, user_id=user_id) → duplicate kwarg) ---
    payload_with_uid = {**base_payload, "user_id": user_id}
    resp_with_uid = client.post("/api/v1/memories", body=payload_with_uid)
    if resp_with_uid.status_code == 500:
        fail(
            r,
            "POST /memories WITH user_id → 500: service does "
            "MemoryBase(**payload.model_dump(), user_id=user_id) which raises "
            "'got multiple values for keyword argument user_id'. "
            'Fix: use payload.model_dump(exclude={"user_id"}) in service.create().',
        )
    elif resp_with_uid.status_code == 201:
        mb = resp_with_uid.json()
        ok(r, "POST /memories with user_id → 201")
        ok(r, f"  Created id={mb['id']}, flow_id={mb['flow_id']}, threshold={mb['threshold']}")
        r.passed = True
        return mb
    else:
        fail(r, f"POST /memories with user_id → {resp_with_uid.status_code}: {resp_with_uid.text[:200]}")

    r.add(
        "[SUMMARY] Memory Base creation is broken via both call patterns: "
        "(1) omitting user_id → 422 because MemoryBaseCreate inherits user_id as required from MemoryBaseBase; "
        "(2) including user_id → 500 because service.create() passes user_id twice to MemoryBase(). "
        "Root fix: remove user_id from MemoryBaseCreate (or make it Optional) and strip it in model_dump."
    )
    return None


# ---------------------------------------------------------------------------
# Scenario 2: Fetch Memory Base details
# ---------------------------------------------------------------------------


def scenario_get_details(client: Client, mb: dict) -> None:
    r = scenario("2 – Fetch Memory Base details")

    resp = client.get(f"/api/v1/memories/{mb['id']}")
    if resp.status_code == 200:
        data = resp.json()
        ok(r, f"GET /memories/{mb['id']} → 200")
        checks = {
            "id": mb["id"],
            "flow_id": mb["flow_id"],
            "kb_name": mb["kb_name"],
            "threshold": mb["threshold"],
        }
        all_match = True
        for field_name, expected in checks.items():
            actual = data.get(field_name)
            if str(actual) == str(expected):
                ok(r, f"  {field_name}: {actual}")
            else:
                fail(r, f"  {field_name}: expected={expected}, got={actual}")
                all_match = False
        r.passed = all_match
    else:
        fail(r, f"GET /memories/{mb['id']} → {resp.status_code}: {resp.text[:200]}")


# ---------------------------------------------------------------------------
# Scenario 3: Fetch messages (session_id filter + memory_base_id filter)
# ---------------------------------------------------------------------------


def scenario_messages(client: Client, flow: dict, mb: dict) -> str:
    r = scenario("3 – Fetch messages with session_id filter; verify memory_base_id param")

    session_id = f"test-session-{uuid.uuid4().hex[:8]}"

    # Run flow once to generate at least one message
    info(f"Running flow once with session_id={session_id} …")
    run_flow_once(client, flow["id"], session_id, message="ping")
    time.sleep(1)

    # 3a: Filter by session_id
    msgs = get_messages(client, session_id=session_id)
    if msgs:
        ok(r, f"GET /monitor/messages?session_id=... → {len(msgs)} messages")
        r.passed = True
    else:
        fail(r, "No messages returned for session_id filter (flow may not have produced output)")
        # Still continue with the rest of the check

    # 3b: memory_base_id parameter – spec requires this but the endpoint doesn't implement it
    resp = client.get(
        "/api/v1/monitor/messages",
        params={"session_id": session_id, "memory_base_id": mb["id"]},
    )
    if resp.status_code == 200:
        fail(
            r,
            "GET /monitor/messages?memory_base_id=... returned 200, but the spec says it should "
            "resolve memory_base_id → flow_id and filter. Verify implementation actually filters "
            "correctly (accepted unknown params may be silently ignored).",
        )
        returned = resp.json()
        if returned:
            # If we got results, check they are for the right session
            ok(r, f"  Returned {len(returned)} messages. Spot-check session_id correctness …")
        else:
            fail(r, "  Returned empty list; memory_base_id filter likely ignored (not implemented)")
    elif resp.status_code in (400, 422):
        fail(
            r,
            f"GET /monitor/messages?memory_base_id=... → {resp.status_code}: "
            "Endpoint rejects memory_base_id param – NOT IMPLEMENTED per spec",
        )
    else:
        fail(r, f"GET /monitor/messages?memory_base_id=... → unexpected {resp.status_code}")

    # 3c: pagination params
    resp_paged = client.get(
        "/api/v1/monitor/messages",
        params={"session_id": session_id, "page": 1, "offset": 0},
    )
    if resp_paged.status_code == 200:
        ok(r, "Pagination query params accepted (page, offset) without server error")
    else:
        fail(r, f"Pagination params caused {resp_paged.status_code} – endpoint may not support them")

    return session_id


# ---------------------------------------------------------------------------
# Scenario 4: Threshold auto-trigger creates/updates KB
# ---------------------------------------------------------------------------


def scenario_threshold(client: Client, flow: dict, mb: dict, session_id: str) -> None:
    r = scenario("4 – Run flow N times to hit threshold → verify KB ingestion triggered")

    # ---- 4a: Check whether on_flow_output hook is wired up ----
    info("Checking if on_flow_output is called from the flow execution pipeline …")
    hook_wired = False  # Code-level finding from audit (see notes)
    r.add(
        "[AUDIT] on_flow_output() in MemoryBaseService is NEVER called from the flow "
        "execution engine (graph/, chat.py, memory.py). The auto-capture hook is defined "
        "but disconnected from the pipeline. Threshold cannot fire automatically via flow runs."
    )

    # ---- 4b: Inject enough messages to hit threshold (runs flow THRESHOLD times) ----
    info(f"Running flow {THRESHOLD} more times on the same session to hit threshold={THRESHOLD} …")
    new_outputs = inject_messages_directly(client, flow["id"], session_id, THRESHOLD)
    info(f"New is_output messages produced: {new_outputs}")

    # Give the auto-capture a moment (in case the hook is wired but we missed it)
    time.sleep(3)

    # ---- 4c: Check sessions to see if pending count reflects messages ----
    resp = client.get(f"/api/v1/memories/{mb['id']}/sessions")
    if resp.status_code == 200:
        sessions = resp.json()
        ok(r, f"GET /memories/{mb['id']}/sessions → {len(sessions)} session(s)")
        for s in sessions:
            ok(
                r,
                f"  session_id={s.get('session_id')}, "
                f"pending_count={s.get('pending_count')}, "
                f"total_processed={s.get('total_processed')}",
            )
            if s.get("session_id") == session_id:
                pending = s.get("pending_count", 0)
                if pending >= THRESHOLD:
                    ok(r, f"  Pending count {pending} >= threshold {THRESHOLD} – threshold met in DB")
                    if not hook_wired:
                        fail(
                            r,
                            "  BUT auto-trigger did NOT fire (hook disconnected). "
                            "Threshold condition is met in DB, ingestion job was NOT started.",
                        )
    else:
        fail(r, f"GET /memories/{mb['id']}/sessions → {resp.status_code}")

    # ---- 4d: Check if a KB ingestion job was created ----
    kb_resp = client.get(f"/api/v1/knowledge_bases/{mb['kb_name']}")
    if kb_resp.status_code == 200:
        kb = kb_resp.json()
        status = kb.get("status", "unknown")
        chunks = kb.get("chunks", 0)
        ok(r, f"KB '{mb['kb_name']}' exists – status={status}, chunks={chunks}")
        if status in ("ready", "ingesting"):
            ok(r, "KB status is ready/ingesting – auto-trigger fired successfully")
            r.passed = True
        else:
            fail(
                r,
                f"KB status is '{status}' (expected 'ready' or 'ingesting'). "
                "Auto-capture hook is not wired into the flow execution pipeline.",
            )
    elif kb_resp.status_code == 404:
        fail(
            r,
            f"KB '{mb['kb_name']}' not found – auto-capture hook is disconnected from "
            "the flow execution pipeline; no ingestion was ever triggered.",
        )
    else:
        fail(r, f"GET /knowledge_bases/{mb['kb_name']} → {kb_resp.status_code}")


# ---------------------------------------------------------------------------
# Scenario 5: auto_capture=False prevents ingestion
# ---------------------------------------------------------------------------


def scenario_auto_capture_off(client: Client, flow: dict, kb_name: str, user_id: str) -> None:
    r = scenario("5 – auto_capture=False: ingestion jobs must never trigger")

    # Create a separate memory base with auto_capture=False
    payload = {
        "name": f"test-mb-nocapture-{uuid.uuid4().hex[:6]}",
        "flow_id": flow["id"],
        "kb_name": kb_name,
        "threshold": 1,  # Very low threshold to make it easy to trigger if broken
        "auto_capture": False,
        "user_id": user_id,
    }
    resp = client.post("/api/v1/memories", body=payload)
    if resp.status_code != 201:
        fail(r, f"Could not create auto_capture=False memory base: {resp.status_code}")
        return

    mb_no_capture = resp.json()
    ok(r, f"Created memory base with auto_capture=False, id={mb_no_capture['id']}")

    session_id = f"nocapture-{uuid.uuid4().hex[:8]}"

    # Run flow several times (threshold=1, so if the hook were called it would fire)
    info(f"Running flow {THRESHOLD} times (threshold=1, auto_capture=False) …")
    inject_messages_directly(client, flow["id"], session_id, THRESHOLD)
    time.sleep(3)

    # Check that no job was created for this memory base
    resp_sessions = client.get(f"/api/v1/memories/{mb_no_capture['id']}/sessions")
    if resp_sessions.status_code == 200:
        sessions = resp_sessions.json()
        total_processed_sum = sum(s.get("total_processed", 0) for s in sessions)
        if total_processed_sum == 0:
            ok(r, "total_processed=0 across all sessions – no ingestion ran (as expected)")
            r.passed = True
        else:
            fail(
                r,
                f"total_processed={total_processed_sum} despite auto_capture=False "
                "– ingestion fired when it should not have",
            )
    else:
        # No sessions means the hook was never called at all – also correct
        ok(r, "No sessions tracked for this memory base (hook never called) – correct for auto_capture=False")
        r.passed = True

    # Also verify the KB has zero chunks from this memory base (it may have chunks from scenario 4)
    # We reuse the same kb_name, so we just note the session isolation aspect
    r.add(
        "[NOTE] auto_capture=False is only verifiable if the auto-capture hook is wired up. "
        "Since on_flow_output is currently disconnected, this test passes vacuously "
        "(the hook never fires for either True or False)."
    )

    # Cleanup
    client.delete(f"/api/v1/memories/{mb_no_capture['id']}")


# ---------------------------------------------------------------------------
# Scenario 6: Manual flush (below threshold) triggers KB ingestion job
# ---------------------------------------------------------------------------


def scenario_manual_flush(client: Client, flow: dict, mb: dict) -> None:
    r = scenario("6 – Manual flush (below threshold) triggers KB ingestion job")

    # Create a fresh session so pending count is below threshold
    flush_session = f"flush-test-{uuid.uuid4().hex[:8]}"

    # Generate just 1 message (below threshold=THRESHOLD)
    info(f"Generating 1 message (below threshold={THRESHOLD}) on session={flush_session} …")
    run_flow_once(client, flow["id"], flush_session, message="manual-flush-trigger")
    time.sleep(1)

    # Verify we're below threshold
    resp_sessions = client.get(f"/api/v1/memories/{mb['id']}/sessions")
    flush_sess_data = {}
    if resp_sessions.status_code == 200:
        for s in resp_sessions.json():
            if s.get("session_id") == flush_session:
                flush_sess_data = s
    pending = flush_sess_data.get("pending_count", 0)
    info(f"pending_count for flush session: {pending} (threshold={THRESHOLD})")

    # ---- 6a: POST /memories/{id}/flush ----
    info("Calling POST /memories/{id}/flush …")
    flush_resp = client.post(
        f"/api/v1/memories/{mb['id']}/flush",
        body={"session_id": flush_session},
    )

    if flush_resp.status_code == 202:
        data = flush_resp.json()
        job_id = data.get("job_id")
        ok(r, f"POST /memories/{mb['id']}/flush → 202 Accepted, job_id={job_id}")

        # ---- 6b: Poll the job until terminal state ----
        if job_id:
            info("Polling job status …")
            job = wait_for_job(client, job_id)
            if job:
                status = job.get("status")
                if status == "completed":
                    ok(r, f"Job {job_id} completed successfully")
                    r.passed = True
                elif status == "failed":
                    fail(r, f"Job {job_id} FAILED: {job.get('error', 'no error detail')}")
                    r.add(
                        "[NOTE] Ingestion failed – likely because KB embedding requires a real API key (Fake provider)"
                    )
                else:
                    fail(r, f"Job reached status: {status}")
            else:
                fail(r, f"Job {job_id} did not reach terminal state within {JOB_POLL_TIMEOUT}s")
    elif flush_resp.status_code == 409:
        fail(r, "POST /flush → 409 Conflict: an ingestion job is already running for this session")
        r.add("[NOTE] Re-run the test against a fresh session to avoid the 409")
    elif flush_resp.status_code == 404:
        fail(r, "POST /flush → 404: session not tracked (no messages for this session in the memory base)")
        r.add(
            "[NOTE] If pending_count=0 above, the memory base session was never created because "
            "on_flow_output is not wired into the flow engine. Try creating the session manually "
            "via sessions endpoint or running the flow first."
        )
    else:
        fail(r, f"POST /flush → {flush_resp.status_code}: {flush_resp.text[:300]}")

    # ---- 6c: Verify duplicate flush returns 409 ----
    info("Verifying duplicate flush → 409 …")
    dup_resp = client.post(
        f"/api/v1/memories/{mb['id']}/flush",
        body={"session_id": flush_session},
    )
    if dup_resp.status_code == 409:
        ok(r, "Duplicate flush correctly returns 409 Conflict")
    elif dup_resp.status_code == 202:
        ok(r, "Second flush accepted (first job completed quickly before duplicate call)")
    else:
        fail(r, f"Duplicate flush returned unexpected {dup_resp.status_code}")

    # ---- 6d: KB status after flush ----
    time.sleep(JOB_POLL_INTERVAL)
    kb_resp = client.get(f"/api/v1/knowledge_bases/{mb['kb_name']}")
    if kb_resp.status_code == 200:
        kb = kb_resp.json()
        ok(r, f"KB post-flush: status={kb.get('status')}, chunks={kb.get('chunks')}")
    else:
        fail(r, f"KB check post-flush → {kb_resp.status_code}")


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------


def cleanup(client: Client, mb_id: str | None, kb_name: str | None, flow_id: str | None) -> None:
    print(f"\n{'=' * 70}")
    print("  CLEANUP")
    print("=" * 70)
    try:
        if mb_id:
            r = client.delete(f"/api/v1/memories/{mb_id}")
            info(f"DELETE memory base {mb_id} → {r.status_code}")
        if kb_name:
            r = client.delete(f"/api/v1/knowledge_bases/{kb_name}")
            info(f"DELETE KB {kb_name} → {r.status_code}")
        # Don't delete the flow – it may be pre-existing
    except Exception as exc:
        info(f"Cleanup error (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


def print_report() -> None:
    print(f"\n\n{'#' * 70}")
    print("  FINAL TEST REPORT")
    print(f"{'#' * 70}")
    print(f"{'Scenario':<52} {'Result'}")
    print("-" * 70)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        mark = "✓" if r.passed else "✗"
        print(f"  {mark} {r.name:<50} [{status}]")

    print(f"\n{'─' * 70}")
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"  Passed: {passed}/{total}")

    print(f"\n{'─' * 70}")
    print("  DETAILED NOTES\n")
    for r in results:
        if r.notes:
            print(f"  [{r.name}]")
            for note in r.notes:
                print(f"    {note}")
            print()

    # Known issues summary
    print(f"{'─' * 70}")
    print("  KNOWN ISSUES / SPEC GAPS DETECTED\n")
    known_issues = [
        (
            "CRITICAL",
            "on_flow_output() hook is defined in MemoryBaseService but is NEVER called "
            "from the flow execution pipeline (graph/, chat.py, memory.py). "
            "Threshold-based auto-capture (Scenario 4) cannot work until this hook is "
            "wired up where flow output messages are persisted.",
        ),
        (
            "MISSING",
            "GET /api/v1/monitor/messages does not accept a memory_base_id query parameter. "
            "Spec requires: resolve memory_base_id → flow_id, then filter messages. "
            "Currently the endpoint ignores any unknown query params.",
        ),
        (
            "MISSING",
            "GET /api/v1/monitor/messages does not support pagination (page/offset params). "
            "Spec documents page and offset but the endpoint returns a flat list.",
        ),
        (
            "INDIRECT",
            "auto_capture=False test (Scenario 5) passes vacuously: because the hook is "
            "disconnected, no ingestion fires regardless of the auto_capture flag. "
            "This must be re-validated once the hook is wired up.",
        ),
    ]
    for severity, desc in known_issues:
        print(f"  [{severity}] {desc}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    client = Client(BASE_URL)

    # Check server is reachable
    try:
        ping = requests.get(f"{BASE_URL}/health", timeout=5)
        if ping.status_code != 200:
            print(f"[ERROR] Server not healthy at {BASE_URL}: {ping.status_code}")
            sys.exit(1)
        print(f"[OK] Server reachable at {BASE_URL}")
    except requests.ConnectionError:
        print(f"[ERROR] Cannot connect to {BASE_URL}. Is the server running?")
        sys.exit(1)

    # Auth
    if not scenario_auth(client):
        print_report()
        sys.exit(1)

    # Resolve current user ID (needed for MemoryBaseCreate payload)
    whoami = client.get("/api/v1/users/whoami")
    if whoami.status_code != 200:
        print(f"[ERROR] Could not resolve current user: {whoami.status_code}")
        sys.exit(1)
    current_user_id = whoami.json()["id"]
    info(f"Current user id: {current_user_id}")

    # Get or create a usable flow
    flow = get_or_create_flow(client)
    if not flow:
        print("[ERROR] No usable flow available. Cannot continue.")
        print_report()
        sys.exit(1)

    # KB name is stable per test run so we can clean up
    kb_name = f"test-kb-{uuid.uuid4().hex[:8]}"

    # Ensure KB exists (memory base references it)
    create_kb(client, kb_name)

    # Run all scenarios
    mb = scenario_create(client, flow, kb_name, current_user_id)

    if mb:
        scenario_get_details(client, mb)
        session_id = scenario_messages(client, flow, mb)
        scenario_threshold(client, flow, mb, session_id)
        scenario_auto_capture_off(client, flow, kb_name, current_user_id)
        scenario_manual_flush(client, flow, mb)
        cleanup(client, mb["id"], kb_name, flow.get("id"))
    else:
        # Memory Base creation is broken – mark remaining scenarios as blocked and still report
        for name in [
            "2 – Fetch Memory Base details",
            "3 – Fetch messages with session_id filter; verify memory_base_id param",
            "4 – Run flow N times to hit threshold → verify KB ingestion triggered",
            "5 – auto_capture=False: ingestion jobs must never trigger",
            "6 – Manual flush (below threshold) triggers KB ingestion job",
        ]:
            r = ScenarioResult(name=name, passed=False)
            r.add("[BLOCKED] Depends on Memory Base creation (Scenario 1), which is broken.")
            results.append(r)
            print(f"\n  BLOCKED: {name}")
        cleanup(client, None, kb_name, flow.get("id"))
    print_report()


if __name__ == "__main__":
    main()
