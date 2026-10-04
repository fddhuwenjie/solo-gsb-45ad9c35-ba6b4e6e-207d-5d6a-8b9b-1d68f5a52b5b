"""展厅开放批次合规核验：单事务核验、逐件原因、凭证固化、request_key 幂等与并发。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

BASE = 1_700_000_000  # 2023-11-14T22:13:20Z


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    from app.main import app

    with TestClient(app) as c:
        yield c, db_file


def make_exhibit(client: TestClient, eid: str | None = None,
                 limit: float = 100000.0, interval: float = 3600.0) -> str:
    body = {"name": "唐代绢画", "dose_limit_luxh": limit, "max_interval_seconds": interval}
    if eid is not None:
        body["id"] = eid
    resp = client.post("/exhibits", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["canonical_id"]


def set_calendar_24h(client: TestClient, eid: str) -> None:
    """全天 24 小时开放（00:00–00:00 跨午夜到次日同一时刻）。"""
    resp = client.put(
        f"/exhibits/{eid}/calendar",
        json={
            "timezone": "UTC",
            "weekly": [{"weekday": d, "start": "00:00", "end": "00:00"}
                       for d in range(1, 8)],
        },
    )
    assert resp.status_code == 200, resp.text


def send(client: TestClient, exhibit_id: str, readings: list[dict]) -> dict:
    resp = client.post(f"/exhibits/{exhibit_id}/readings", json={"readings": readings})
    assert resp.status_code == 200, resp.text
    return resp.json()


def reconcile(client: TestClient, round_id: str, expected: list[dict],
              events: list[dict]) -> dict:
    body = {
        "start_at": iso(BASE - 3600),
        "window_seconds": 60,
        "rssi_delta_threshold": 5,
        "expected": expected,
        "events": events,
    }
    resp = client.post(f"/inventory-rounds/{round_id}/reconcile", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def verify(client: TestClient, key: str = "open-1", round_id: str = "R1",
           exhibit_ids: list[str] | None = None, **kw) -> object:
    body = {
        "request_key": key,
        "round_id": round_id,
        "start": iso(BASE),
        "end": iso(BASE + 6 * 3600),
        "exhibit_ids": exhibit_ids if exhibit_ids is not None else ["EX-1"],
        "max_rate_lux_per_second": 1.0,
        "stuck_tolerance_lux": 2.0,
        "min_stuck_seconds": 1800,
    }
    body.update(kw)
    return client.post("/opening-batches/verify", json=body)


def good_exhibit(client: TestClient, eid: str = "EX-1",
                 limit: float = 100000.0) -> str:
    """造一件完全合规的展品：24h 日历 + 窗口内平稳变化的正常读数。"""
    make_exhibit(client, eid, limit=limit)
    set_calendar_24h(client, eid)
    send(client, eid, [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE + 3600), "lux": 110},
        {"id": "r3", "ts": iso(BASE + 7200), "lux": 120},
    ])
    return eid


def located_round(client: TestClient, round_id: str = "R1",
                  pairs: list[tuple[str, str]] | None = None) -> None:
    """固化一轮盘点：pairs 中每件展品都在预期区域被读到。"""
    pairs = pairs if pairs is not None else [("EX-1", "A区")]
    reconcile(
        client, round_id,
        expected=[{"exhibit_id": eid, "area": area} for eid, area in pairs],
        events=[
            {"exhibit_id": eid, "area": area, "ts": iso(BASE - 1800), "rssi": -50}
            for eid, area in pairs
        ],
    )


# ---------- 全部通过：固化凭证 ----------

def test_verify_all_pass_freezes_credential(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    good_exhibit(client, "EX-2")
    located_round(client, pairs=[("EX-1", "A区"), ("EX-2", "B区")])

    resp = verify(client, exhibit_ids=["EX-1", "EX-2"])
    assert resp.status_code == 200, resp.text
    d = resp.json()
    assert d["compliant"] is True
    assert d["already_committed"] is False
    assert d["credential_id"] and len(d["digest"]) == 64
    assert d["created_at"] is not None
    assert len(d["request_digest"]) == 64 and len(d["evidence_digest"]) == 64
    assert [x["requested_id"] for x in d["exhibits"]] == ["EX-1", "EX-2"]
    for x in d["exhibits"]:
        assert x["compliant"] is True and x["failure_reasons"] == []
        ev = x["evidence"]
        assert ev["inventory"] == {
            "status": "located",
            "expected_area": "A区" if x["canonical_id"] == "EX-1" else "B区",
            "located_area": "A区" if x["canonical_id"] == "EX-1" else "B区",
        }
        assert ev["events"] == []
        assert ev["dose"]["exceeded"] is False
        assert ev["dose"]["known_dose_luxh"] == pytest.approx(220.0)  # 105 + 115
        assert ev["open_ticket_ids"] == []
        assert len(ev["readings_fingerprint"]) == 64
        assert len(ev["calendar_fingerprint"]) == 64

    # GET 取回凭证：与固化内容一致
    got = client.get(f"/opening-batches/{d['credential_id']}")
    assert got.status_code == 200
    g = got.json()
    assert g["digest"] == d["digest"]
    assert g["request_digest"] == d["request_digest"]
    assert g["evidence_digest"] == d["evidence_digest"]
    assert g["exhibits"] == d["exhibits"]
    assert g["created_at"] == d["created_at"]
    assert client.get("/opening-batches/no-such").status_code == 404


def test_verify_failure_returns_per_item_reasons_and_writes_nothing(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    # EX-2 有 gap（间隔 2h > 1h 上限）
    make_exhibit(client, "EX-2")
    set_calendar_24h(client, "EX-2")
    send(client, "EX-2", [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE + 7200), "lux": 100},
    ])
    located_round(client, pairs=[("EX-1", "A区"), ("EX-2", "B区")])

    resp = verify(client, exhibit_ids=["EX-1", "EX-2"])
    assert resp.status_code == 200, resp.text
    d = resp.json()
    assert d["compliant"] is False
    assert d["credential_id"] is None and d["digest"] is None
    assert d["created_at"] is None and d["already_committed"] is False
    by_id = {x["canonical_id"]: x for x in d["exhibits"]}
    assert by_id["EX-1"]["compliant"] is True
    assert by_id["EX-1"]["failure_reasons"] == []
    assert by_id["EX-2"]["compliant"] is False
    assert len(by_id["EX-2"]["failure_reasons"]) == 1
    assert "缺测 gap" in by_id["EX-2"]["failure_reasons"][0]
    # 失败不写入：修复后用同一 request_key 重试可以成功固化
    # （补一条 lux 不同的读数消除 gap；lux 差超容差，不会引出 stuck）
    send(client, "EX-2", [{"id": "r3", "ts": iso(BASE + 3600), "lux": 105}])
    retry = verify(client, exhibit_ids=["EX-1", "EX-2"])
    assert retry.status_code == 200
    r = retry.json()
    assert r["compliant"] is True and r["credential_id"] is not None
    assert r["already_committed"] is False


# ---------- 盘点定位核验 ----------

def test_inventory_failures_per_item(env):
    client, _ = env
    for eid in ("EX-1", "EX-2", "EX-3", "EX-4", "EX-5"):
        good_exhibit(client, eid)
    # EX-1 错位、EX-2 缺失、EX-3 位置不明、EX-4 意外出现（不在预期清单）、EX-5 不在轮次中
    reconcile(
        client, "R1",
        expected=[
            {"exhibit_id": "EX-1", "area": "A区"},
            {"exhibit_id": "EX-2", "area": "A区"},
            {"exhibit_id": "EX-3", "area": "B区"},
        ],
        events=[
            {"exhibit_id": "EX-1", "area": "C区", "ts": iso(BASE - 1800), "rssi": -50},
            {"exhibit_id": "EX-3", "area": "B区", "ts": iso(BASE - 1800), "rssi": -50},
            {"exhibit_id": "EX-3", "area": "C区", "ts": iso(BASE - 1790), "rssi": -52},
            {"exhibit_id": "EX-4", "area": "A区", "ts": iso(BASE - 1800), "rssi": -60},
        ],
    )
    resp = verify(client, exhibit_ids=["EX-1", "EX-2", "EX-3", "EX-4", "EX-5"])
    assert resp.status_code == 200, resp.text
    d = resp.json()
    assert d["compliant"] is False
    by_id = {x["canonical_id"]: x for x in d["exhibits"]}
    assert by_id["EX-1"]["evidence"]["inventory"]["status"] == "misplaced"
    assert "错位" in by_id["EX-1"]["failure_reasons"][0]
    assert by_id["EX-2"]["evidence"]["inventory"]["status"] == "missing"
    assert "缺失" in by_id["EX-2"]["failure_reasons"][0]
    assert by_id["EX-3"]["evidence"]["inventory"]["status"] == "unknown_position"
    assert "位置不明" in by_id["EX-3"]["failure_reasons"][0]
    assert by_id["EX-4"]["evidence"]["inventory"]["status"] == "unexpected"
    assert "意外出现" in by_id["EX-4"]["failure_reasons"][0]
    assert by_id["EX-5"]["evidence"]["inventory"]["status"] == "not_expected"
    assert "不在轮次" in by_id["EX-5"]["failure_reasons"][0]


# ---------- 诊断事件核验 ----------

def test_gap_jump_stuck_block_verification(env):
    client, _ = env
    # gap：间隔超过 max_interval_seconds
    make_exhibit(client, "EX-GAP")
    set_calendar_24h(client, "EX-GAP")
    send(client, "EX-GAP", [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE + 7200), "lux": 100},
    ])
    # jump：速率超限
    make_exhibit(client, "EX-JUMP")
    set_calendar_24h(client, "EX-JUMP")
    send(client, "EX-JUMP", [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE + 60), "lux": 1000},
    ])
    # stuck：4 条平稳读数成链
    make_exhibit(client, "EX-STUCK")
    set_calendar_24h(client, "EX-STUCK")
    send(client, "EX-STUCK", [
        {"id": f"r{i}", "ts": iso(BASE + i * 3600), "lux": 100} for i in range(4)
    ])
    located_round(client, pairs=[("EX-GAP", "A区"), ("EX-JUMP", "A区"), ("EX-STUCK", "A区")])

    resp = verify(client, exhibit_ids=["EX-GAP", "EX-JUMP", "EX-STUCK"])
    assert resp.status_code == 200
    d = resp.json()
    assert d["compliant"] is False
    by_id = {x["canonical_id"]: x for x in d["exhibits"]}
    assert [e["type"] for e in by_id["EX-GAP"]["evidence"]["events"]] == ["gap"]
    assert any("缺测 gap" in r for r in by_id["EX-GAP"]["failure_reasons"])
    assert [e["type"] for e in by_id["EX-JUMP"]["evidence"]["events"]] == ["jump"]
    assert any("突变 jump" in r for r in by_id["EX-JUMP"]["failure_reasons"])
    assert [e["type"] for e in by_id["EX-STUCK"]["evidence"]["events"]] == ["stuck"]
    assert any("卡死 stuck" in r for r in by_id["EX-STUCK"]["failure_reasons"])


def test_timestamp_conflict_does_not_block(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    set_calendar_24h(client, "EX-1")
    send(client, "EX-1", [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE + 3600), "lux": 110},
        {"id": "r2b", "ts": iso(BASE + 3600), "lux": 130},  # 同时刻异值 → timestamp_conflict
        {"id": "r3", "ts": iso(BASE + 7200), "lux": 120},
    ])
    located_round(client)
    resp = verify(client)
    assert resp.status_code == 200, resp.text
    d = resp.json()
    # timestamp_conflict 只记入证据，不阻断核验
    assert d["compliant"] is True, d["exhibits"][0]["failure_reasons"]
    events = d["exhibits"][0]["evidence"]["events"]
    assert [e["type"] for e in events] == ["timestamp_conflict"]


# ---------- 开放剂量核验 ----------

def test_dose_exceeded_blocks(env):
    client, _ = env
    make_exhibit(client, "EX-1", limit=50.0)  # 上限很低：1h × 100 lux 即越限
    set_calendar_24h(client, "EX-1")
    send(client, "EX-1", [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE + 3600), "lux": 100},
    ])
    located_round(client)
    resp = verify(client)
    assert resp.status_code == 200
    d = resp.json()
    assert d["compliant"] is False
    x = d["exhibits"][0]
    assert x["evidence"]["dose"]["exceeded"] is True
    assert any("开放剂量越限" in r for r in x["failure_reasons"])


def test_missing_calendar_blocks(env):
    client, _ = env
    make_exhibit(client, "EX-1")  # 不配置日历
    send(client, "EX-1", [{"id": "r1", "ts": iso(BASE), "lux": 100}])
    located_round(client)
    resp = verify(client)
    assert resp.status_code == 200
    d = resp.json()
    assert d["compliant"] is False
    x = d["exhibits"][0]
    assert x["evidence"]["dose"] is None
    assert x["evidence"]["calendar_fingerprint"] is None
    assert any("未配置展陈开放日历" in r for r in x["failure_reasons"])


# ---------- 未结处置单核验 ----------

def _create_ticket(client: TestClient, eid: str, key: str = "t-1") -> dict:
    resp = client.post(
        f"/exhibits/{eid}/illumination-tickets",
        json={
            "start": iso(BASE), "end": iso(BASE + 6 * 3600),
            "max_rate_lux_per_second": 1.0, "stuck_tolerance_lux": 2.0,
            "min_stuck_seconds": 1800, "request_key": key,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_open_ticket_blocks_until_closed(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    located_round(client)
    t = _create_ticket(client, "EX-1")

    resp = verify(client)
    assert resp.status_code == 200
    d = resp.json()
    assert d["compliant"] is False
    x = d["exhibits"][0]
    assert x["evidence"]["open_ticket_ids"] == [t["ticket_id"]]
    assert any("未结照度处置单" in r for r in x["failure_reasons"])

    # 结案（accepted）后同一 request_key 重试成功（失败未占用键）
    close = client.post(
        f"/illumination-tickets/{t['ticket_id']}/close",
        json={"expected_digest": t["digest"], "decision": "accepted",
              "note": "夜间红外补光属正常工况"},
    )
    assert close.status_code == 200, close.text
    retry = verify(client)
    assert retry.status_code == 200
    r = retry.json()
    assert r["compliant"] is True
    assert r["exhibits"][0]["evidence"]["open_ticket_ids"] == []


# ---------- 幂等：重放与 409 ----------

def test_replay_same_request_returns_frozen_credential(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    located_round(client)
    first = verify(client).json()
    assert first["compliant"] is True
    replay = verify(client)
    assert replay.status_code == 200
    r = replay.json()
    assert r["already_committed"] is True
    assert r["credential_id"] == first["credential_id"]
    assert r["digest"] == first["digest"]
    assert r["exhibits"] == first["exhibits"]
    assert r["created_at"] == first["created_at"]


def test_replay_with_equivalent_timezone_and_order(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    good_exhibit(client, "EX-2")
    located_round(client, pairs=[("EX-1", "A区"), ("EX-2", "B区")])
    first = verify(client, exhibit_ids=["EX-1", "EX-2"],
                   start="2023-11-15T06:13:20+08:00",  # = BASE（UTC）
                   end="2023-11-15T12:13:20+08:00").json()
    assert first["compliant"] is True
    # 等价时区写法 + 清单顺序不同：规范化输入一致 → 重放
    replay = verify(client, exhibit_ids=["EX-2", "EX-1"],
                    start=iso(BASE), end=iso(BASE + 6 * 3600))
    assert replay.status_code == 200
    assert replay.json()["already_committed"] is True
    assert replay.json()["credential_id"] == first["credential_id"]


def test_same_key_different_input_409(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    good_exhibit(client, "EX-2")
    located_round(client, pairs=[("EX-1", "A区"), ("EX-2", "B区")])
    assert verify(client, exhibit_ids=["EX-1"]).json()["compliant"] is True
    # 换展品清单 / 窗口 / 阈值 / 轮次 → 409
    assert verify(client, exhibit_ids=["EX-2"]).status_code == 409
    assert verify(client, exhibit_ids=["EX-1"], end=iso(BASE + 8 * 3600)).status_code == 409
    assert verify(client, exhibit_ids=["EX-1"], min_stuck_seconds=900).status_code == 409
    reconcile(client, "R2",
              expected=[{"exhibit_id": "EX-1", "area": "A区"}],
              events=[{"exhibit_id": "EX-1", "area": "A区",
                       "ts": iso(BASE - 1800), "rssi": -50}])
    assert verify(client, exhibit_ids=["EX-1"], round_id="R2").status_code == 409


def test_replay_409_after_state_changes(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    located_round(client)
    first = verify(client).json()
    assert first["compliant"] is True

    # 新上报读数 → 证据摘要变化 → 重放 409
    send(client, "EX-1", [{"id": "r9", "ts": iso(BASE + 10800), "lux": 130}])
    assert verify(client).status_code == 409

    # 作废批次 → 409
    good_exhibit(client, "EX-9")
    located_round(client, "R9", pairs=[("EX-9", "A区")])
    assert verify(client, key="k9", round_id="R9", exhibit_ids=["EX-9"]).json()["compliant"]
    receipt_id = client.get("/exhibits/EX-9/receipts").json()["receipts"][0]["receipt_id"]
    fix = client.post(
        "/exhibits/EX-9/receipts/corrections",
        json={"receipt_id": receipt_id, "request_key": "void-9", "reason": "漂移作废"},
    )
    assert fix.status_code == 200
    assert verify(client, key="k9", round_id="R9", exhibit_ids=["EX-9"]).status_code == 409

    # 修改日历 → 409
    good_exhibit(client, "EX-8")
    located_round(client, "R8", pairs=[("EX-8", "A区")])
    assert verify(client, key="k8", round_id="R8", exhibit_ids=["EX-8"]).json()["compliant"]
    client.put(
        "/exhibits/EX-8/calendar",
        json={"timezone": "UTC",
              "weekly": [{"weekday": d, "start": "09:00", "end": "17:00"}
                         for d in range(1, 8)]},
    )
    assert verify(client, key="k8", round_id="R8", exhibit_ids=["EX-8"]).status_code == 409

    # 发布来源优先级版本 → 409
    good_exhibit(client, "EX-7")
    located_round(client, "R7", pairs=[("EX-7", "A区")])
    assert verify(client, key="k7", round_id="R7", exhibit_ids=["EX-7"]).json()["compliant"]
    sp = client.put(
        "/exhibits/EX-7/source-priority",
        json={"effective_at": iso(BASE), "priorities": ["sensor-a"], "expected_version": 0},
    )
    assert sp.status_code == 201
    assert verify(client, key="k7", round_id="R7", exhibit_ids=["EX-7"]).status_code == 409

    # 新开未结处置单 → 409
    good_exhibit(client, "EX-6")
    located_round(client, "R6", pairs=[("EX-6", "A区")])
    assert verify(client, key="k6", round_id="R6", exhibit_ids=["EX-6"]).json()["compliant"]
    _create_ticket(client, "EX-6", key="t-6")
    assert verify(client, key="k6", round_id="R6", exhibit_ids=["EX-6"]).status_code == 409

    # 首次凭证本身不受影响：仍可按 credential_id 取回
    got = client.get(f"/opening-batches/{first['credential_id']}")
    assert got.status_code == 200
    assert got.json()["digest"] == first["digest"]


def test_replay_409_after_merge_changes_readings(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    located_round(client)
    assert verify(client).json()["compliant"] is True
    # 归并把另一档案的读数搬进 EX-1：读数集合变化 → 证据摘要不一致 → 409
    make_exhibit(client, "DUP-1")
    send(client, "DUP-1", [{"id": "d1", "ts": iso(BASE + 1800), "lux": 105}])
    prev = client.post("/merges/preview", json={"source_id": "DUP-1", "target_id": "EX-1"})
    assert prev.status_code == 200
    commit = client.post("/merges/commit",
                         json={"token": prev.json()["token"], "resolutions": []})
    assert commit.status_code == 200
    assert verify(client).status_code == 409


# ---------- 别名与错误码 ----------

def test_verify_via_alias_after_merge(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    make_exhibit(client, "DUP-1")
    prev = client.post("/merges/preview", json={"source_id": "DUP-1", "target_id": "EX-1"})
    commit = client.post("/merges/commit",
                         json={"token": prev.json()["token"], "resolutions": []})
    assert commit.status_code == 200
    located_round(client)  # 轮次预期清单用规范 ID EX-1
    # 核验清单用别名：解析到规范档案
    resp = verify(client, exhibit_ids=["DUP-1"])
    assert resp.status_code == 200, resp.text
    d = resp.json()
    assert d["compliant"] is True
    x = d["exhibits"][0]
    assert x["requested_id"] == "DUP-1" and x["canonical_id"] == "EX-1"
    # 别名重放同一 request_key（解析到同一规范档案）→ 回放
    replay = verify(client, exhibit_ids=["DUP-1"])
    assert replay.json()["already_committed"] is True
    # 规范 ID 与别名混用 → 解析后重复 → 422
    assert verify(client, key="k-dup", exhibit_ids=["EX-1", "DUP-1"]).status_code == 422


def test_unknown_exhibit_and_round_404(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    located_round(client)
    assert verify(client, exhibit_ids=["no-such"]).status_code == 404
    assert verify(client, round_id="no-such-round").status_code == 404


def test_validation_422(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    located_round(client)
    # end 不晚于 start
    assert verify(client, start=iso(BASE), end=iso(BASE)).status_code == 422
    # 清单重复（同一 ID 出现两次）
    assert verify(client, exhibit_ids=["EX-1", "EX-1"]).status_code == 422
    # 空清单 / 非法阈值（pydantic）
    assert verify(client, exhibit_ids=[]).status_code == 422
    assert verify(client, max_rate_lux_per_second=0).status_code == 422


# ---------- 并发 ----------

def test_concurrent_same_request_key_single_credential(env):
    client, _ = env
    good_exhibit(client, "EX-1")
    located_round(client)
    ids: list[str] = []
    replays: list[bool] = []

    def post(_: int) -> None:
        r = verify(client, key="conc-key")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["compliant"] is True
        ids.append(body["credential_id"])
        replays.append(body["already_committed"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(post, range(8)))
    assert len(set(ids)) == 1
    assert replays.count(False) == 1 and replays.count(True) == 7


# ---------- 与既有接口口径一致 ----------

def test_evidence_matches_diagnose_and_dose_endpoints(env):
    client, _ = env
    # stuck 展品：核验证据中的事件与剂量应和既有接口口径一致
    make_exhibit(client, "EX-1")
    set_calendar_24h(client, "EX-1")
    send(client, "EX-1", [
        {"id": f"r{i}", "ts": iso(BASE + i * 3600), "lux": 100} for i in range(4)
    ])
    located_round(client)
    resp = verify(client)
    d = resp.json()
    assert d["compliant"] is False
    ev = d["exhibits"][0]["evidence"]

    diag = client.post(
        "/exhibits/EX-1/readings/diagnose",
        json={"start": iso(BASE), "end": iso(BASE + 6 * 3600),
              "max_rate_lux_per_second": 1.0, "stuck_tolerance_lux": 2.0,
              "min_stuck_seconds": 1800},
    ).json()
    # 诊断接口以 exclude_unset 序列化（None 字段不出现）；核验证据与处置单快照
    # 一样保留 null 键，比较前剔除
    ev_events = [{k: v for k, v in e.items() if v is not None} for e in ev["events"]]
    assert ev_events == diag["events"]

    dose = client.get(
        "/exhibits/EX-1/dose",
        params={"start": iso(BASE), "end": iso(BASE + 6 * 3600), "calendar": "display"},
    ).json()
    assert ev["dose"]["known_dose_luxh"] == pytest.approx(dose["known_dose_luxh"])
    assert ev["dose"]["remaining_luxh"] == pytest.approx(dose["remaining_luxh"])
    assert ev["dose"]["exceeded"] == dose["exceeded"]
    assert ev["dose"]["closed_seconds"] == pytest.approx(dose["closed_seconds"])
