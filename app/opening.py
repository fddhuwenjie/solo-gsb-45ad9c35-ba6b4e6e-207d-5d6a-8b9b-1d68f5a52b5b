"""展厅开放批次合规核验：单事务核验、逐件原因、合规凭证固化与幂等重放。

设计要点：
- 调用方提交 ``request_key``、核验时间窗、盘点 ``round_id``、展品 ID 清单与
  诊断阈值。服务在**一个 SQLite 写事务**（``BEGIN IMMEDIATE``）内完成：
  别名解析 → 读取该轮次固化对账结果 → 逐件按未作废权威时间线重算完整诊断
  事件（与 ``POST /readings/diagnose`` 同口径）与截至窗口末的
  ``calendar=display`` 剂量（与 ``GET /dose?calendar=display`` 同口径）→
  读取未结照度处置单状态。
- **全部通过才固化**：每件展品都须满足——在盘点轮次中定位到预期区域、
  窗口内无 gap / jump / stuck 诊断事件、开放剂量未越限、没有未结（open）
  处置单——才固化包含各项证据摘要的合规凭证；任一失败返回逐件原因
  （``compliant=false``）且**不写入**，``request_key`` 不被占用，处置后可用
  同一键重试。
- **幂等重放**：``request_key`` 与**规范化输入摘要**（时间 UTC、别名解析后
  的规范档案清单、轮次、窗口、阈值）和**当时证据摘要**（固化时的逐件证据）
  绑定。重放时两者都一致才返回首次固化的凭证
  （``already_committed=true``）；输入不同、或读数 / 作废 / 来源优先级 /
  日历 / 处置单 / 归并归属变化导致证据摘要重算不一致 → **409**。
- 凭证只增不改：落库后绝不 UPDATE，不随档案归并级联删除，供事后核查。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException

from .arbitration import (
    arbitrate,
    diagnosis_breaks,
    diagnosis_feed,
    load_priority_versions,
)
from .calendar import CalendarConfig, load_calendar, open_intervals
from .diagnose import DiagnosisEvent, build_events, intersect_window, sort_events
from .dose import compute_dose
from .merge import resolve_canonical
from .receipts import content_hash_of
from .schemas import (
    OpeningBatchCredentialOut,
    OpeningBatchVerifyRequest,
    OpeningBatchVerifyOut,
    ensure_aware,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _round6(x: float) -> float:
    return round(x, 6)


# 不合规的诊断事件类型（timestamp_conflict 只记入证据，不阻断核验）
_BLOCKING_EVENT_TYPES = {"gap": "缺测 gap", "jump": "突变 jump", "stuck": "卡死 stuck"}


# ---------- 证据序列化（与处置单快照同口径，浮点保留 6 位） ----------

def _event_dict(e: DiagnosisEvent) -> dict:
    """诊断事件的证据序列化（与诊断接口输出字段一致，浮点保留 6 位）。"""
    d: dict = {
        "type": e.type,
        "start": _to_iso(e.start),
        "end": _to_iso(e.end),
        "reading_ids": list(e.reading_ids),
    }
    if e.interval_seconds is not None:
        d["interval_seconds"] = _round6(e.interval_seconds)
    if e.rate_lux_per_second is not None:
        d["rate_lux_per_second"] = _round6(e.rate_lux_per_second)
    if e.duration_seconds is not None:
        d["duration_seconds"] = _round6(e.duration_seconds)
    if e.lux_span is not None:
        d["lux_span"] = _round6(e.lux_span)
    return d


def _calendar_payload(cfg: CalendarConfig) -> dict:
    """展陈日历的稳定指纹载荷：时区 + 每周时段 + 例外日（全部按内容排序）。"""
    return {
        "timezone": cfg.timezone,
        "weekly": [
            {"weekday": wd, "start_sec": s, "end_sec": e}
            for wd in sorted(cfg.weekly)
            for s, e in sorted(cfg.weekly[wd])
        ],
        "exceptions": [
            {"date": day.isoformat(),
             "windows": [{"start_sec": s, "end_sec": e} for s, e in sorted(wins)]}
            for day, wins in sorted(cfg.exceptions.items())
        ],
    }


# ---------- 规范化输入摘要与证据摘要 ----------

def _request_digest(
    round_id: str,
    start_s: float,
    end_s: float,
    payload: OpeningBatchVerifyRequest,
    canonical_ids: list[str],
) -> str:
    """规范化输入摘要（SHA-256）。

    规范化口径：时间统一为 UTC ``Z`` 结尾 ISO 8601；展品 ID 一律为别名解析后
    的规范档案 ID、排序去重（顺序与别名写法不影响摘要）；浮点经 receipts 的
    稳定序列化（``float.hex``）参与哈希。
    """
    return content_hash_of({
        "round_id": round_id,
        "window": {"start": _to_iso(start_s), "end": _to_iso(end_s)},
        "thresholds": {
            "max_rate_lux_per_second": payload.max_rate_lux_per_second,
            "stuck_tolerance_lux": payload.stuck_tolerance_lux,
            "min_stuck_seconds": payload.min_stuck_seconds,
        },
        "exhibits": sorted(canonical_ids),
    })


def _evidence_digest(items: list[dict]) -> str:
    """证据摘要（SHA-256）：逐件证据按规范档案 ID 排序后取规范序列化哈希。"""
    return content_hash_of({
        "exhibits": sorted(items, key=lambda x: x["canonical_id"]),
    })


# ---------- 逐件证据计算 ----------

def _inventory_evidence(round_result: dict, canonical_id: str) -> dict:
    """从盘点轮次固化结果中取出一件展品的定位证据。"""
    expected = {e["canonical_id"]: e["expected_area"] for e in round_result["expectations"]}
    missing = {m["canonical_id"] for m in round_result["missing"]}
    misplaced = {m["canonical_id"]: m["actual_area"] for m in round_result["misplaced"]}
    unexpected = {u["canonical_id"]: u["actual_area"] for u in round_result["unexpected"]}
    unknown = {u["canonical_id"] for u in round_result["unknown_position"]}
    if canonical_id in missing:
        return {"status": "missing", "expected_area": expected[canonical_id],
                "located_area": None}
    if canonical_id in misplaced:
        return {"status": "misplaced", "expected_area": expected[canonical_id],
                "located_area": misplaced[canonical_id]}
    if canonical_id in unknown:
        return {"status": "unknown_position", "expected_area": expected.get(canonical_id),
                "located_area": None}
    if canonical_id in unexpected:
        return {"status": "unexpected", "expected_area": None,
                "located_area": unexpected[canonical_id]}
    if canonical_id in expected:
        return {"status": "located", "expected_area": expected[canonical_id],
                "located_area": expected[canonical_id]}
    return {"status": "not_expected", "expected_area": None, "located_area": None}


def _compute_exhibit_evidence(
    conn: sqlite3.Connection,
    canonical_id: str,
    start_s: float,
    end_s: float,
    payload: OpeningBatchVerifyRequest,
) -> dict:
    """按当前库内状态重算一件展品的核验证据（创建固化与重放核对共用同一口径）。

    - 诊断事件：在规范档案的**完整未作废时间线**上形成（已配置来源优先级
      版本时先仲裁为权威时间线，保持冲突的时刻作为断点），再筛选与窗口
      相交的事件——与诊断接口一致；
    - 剂量：``calendar=display`` 口径、截至窗口末累计，与剂量接口一致；
      未配置日历时不算剂量（``dose`` 为 None），由调用方记为不合规；
    - 另含未结处置单、未作废读数指纹、来源优先级配置指纹与日历指纹，
      使读数、作废、来源优先级、日历或处置单的任何变化都改变证据摘要。
    """
    exhibit = conn.execute(
        "SELECT * FROM exhibits WHERE id = ?", (canonical_id,)
    ).fetchone()
    if exhibit is None:
        # 规范档案已不存在（核验与重算之间不会发生，仅兜底）
        raise HTTPException(
            status_code=409,
            detail=f"规范档案 {canonical_id} 已不存在，无法重算核验证据",
        )

    rows = conn.execute(
        "SELECT reading_id, ts, lux, source_id FROM readings "
        "WHERE exhibit_id = ? AND voided_at IS NULL ORDER BY ts, reading_id",
        (canonical_id,),
    ).fetchall()

    # 与诊断/剂量接口一致：配置了来源优先级版本时先逐时刻仲裁，
    # 保持冲突的时刻作为断点（不进入诊断输入与剂量积分）。
    versions = load_priority_versions(conn, canonical_id)
    breaks: list[float] = []
    dose_breaks: list[float] = []
    if versions:
        points, arb_conflicts = arbitrate(
            [(r["reading_id"], r["ts"], r["lux"], r["source_id"]) for r in rows],
            versions,
        )
        feed = diagnosis_feed(points)
        breaks = diagnosis_breaks(arb_conflicts)
        dose_readings = [(p.ts, p.lux) for p in points]
        dose_breaks = [c.ts for c in arb_conflicts]
    else:
        feed = [(r["reading_id"], r["ts"], r["lux"]) for r in rows]
        # 与剂量接口一致：截至窗口末的全部读数 + 窗口后最近一条（参与末端裁剪）
        dose_readings = [(r["ts"], r["lux"]) for r in rows if r["ts"] <= end_s]
        after = [r for r in rows if r["ts"] > end_s]
        if after:
            dose_readings.append((after[0]["ts"], after[0]["lux"]))

    events = build_events(
        feed,
        max_interval_s=exhibit["max_interval_seconds"],
        max_rate_lux_per_s=payload.max_rate_lux_per_second,
        stuck_tol_lux=payload.stuck_tolerance_lux,
        min_stuck_s=payload.min_stuck_seconds,
        breaks=breaks,
    )
    events = sort_events(intersect_window(events, start_s, end_s))

    cfg = load_calendar(conn, canonical_id)
    dose_dict: dict | None = None
    calendar_fingerprint: str | None = None
    if cfg is not None:
        # 开放区间须覆盖截至窗口末的开放历史（与剂量接口同口径）
        first_ts = dose_readings[0][0] if dose_readings else start_s
        intervals = open_intervals(cfg, min(start_s, first_ts), end_s)
        dose = compute_dose(
            readings=dose_readings,
            start_s=start_s,
            end_s=end_s,
            max_interval_s=exhibit["max_interval_seconds"],
            dose_limit_luxh=exhibit["dose_limit_luxh"],
            open_intervals=intervals,
            break_ts=dose_breaks,
        )
        dose_dict = {
            "dose_limit_luxh": exhibit["dose_limit_luxh"],
            "known_dose_luxh": _round6(dose.known_dose_luxh),
            "remaining_luxh": _round6(dose.remaining_luxh),
            "exceeded": dose.exceeded,
            "first_exceeded_at": (
                _to_iso(dose.first_exceeded_at)
                if dose.first_exceeded_at is not None else None
            ),
            "unknown_seconds": _round6(dose.unknown_seconds),
            "covered_seconds": _round6(dose.covered_seconds),
            "closed_seconds": (
                _round6(dose.closed_seconds) if dose.closed_seconds is not None else 0.0
            ),
            "reading_count": dose.reading_count,
        }
        calendar_fingerprint = content_hash_of(_calendar_payload(cfg))

    open_tickets = conn.execute(
        "SELECT ticket_id FROM illumination_tickets "
        "WHERE canonical_id = ? AND status = 'open' ORDER BY created_at, ticket_id",
        (canonical_id,),
    ).fetchall()

    return {
        "events": [_event_dict(e) for e in events],
        "dose": dose_dict,
        "open_ticket_ids": [t["ticket_id"] for t in open_tickets],
        "readings_fingerprint": content_hash_of({
            "readings": [
                [r["reading_id"], _to_iso(r["ts"]), r["lux"], r["source_id"]]
                for r in rows
            ],
        }),
        "source_priority_fingerprint": content_hash_of({
            "versions": [
                {"version": v.version, "effective_at": v.effective_at,
                 "priorities": list(v.priorities)}
                for v in versions
            ],
        }),
        "calendar_fingerprint": calendar_fingerprint,
    }


def _failure_reasons(round_id: str, inventory: dict, evidence: dict) -> list[str]:
    """逐件不合规原因：盘点定位、窗口内诊断事件、开放剂量、未结处置单。"""
    reasons: list[str] = []
    status = inventory["status"]
    if status == "missing":
        reasons.append(
            f"盘点缺失：在轮次 {round_id} 中全程无有效读头事件"
            f"（预期区域 {inventory['expected_area']}）"
        )
    elif status == "misplaced":
        reasons.append(
            f"盘点错位：预期区域 {inventory['expected_area']}，"
            f"实际定位到 {inventory['located_area']}"
        )
    elif status == "unknown_position":
        reasons.append("盘点位置不明：最后一个有事件窗口内多区域信号差不足阈值")
    elif status == "unexpected":
        reasons.append(
            f"盘点意外出现：不在轮次 {round_id} 的预期清单中，"
            f"但被定位到区域 {inventory['located_area']}"
        )
    elif status == "not_expected":
        reasons.append(f"不在轮次 {round_id} 的预期清单中，且全程无有效读头事件")
    for e in evidence["events"]:
        label = _BLOCKING_EVENT_TYPES.get(e["type"])
        if label is not None:
            reasons.append(f"窗口内存在{label}事件（{e['start']} 至 {e['end']}）")
    dose = evidence["dose"]
    if dose is None:
        reasons.append("未配置展陈开放日历，无法核算 calendar=display 开放剂量")
    elif dose["exceeded"]:
        reasons.append(
            f"开放剂量越限：截至窗口末累计开放剂量已达上限"
            f"（首次越线 {dose['first_exceeded_at']}）"
        )
    if evidence["open_ticket_ids"]:
        reasons.append(
            f"存在 {len(evidence['open_ticket_ids'])} 张未结照度处置单："
            + "、".join(evidence["open_ticket_ids"])
        )
    return reasons


# ---------- 别名解析 ----------

def _resolve_exhibits(
    conn: sqlite3.Connection, exhibit_ids: list[str]
) -> list[tuple[str, str]]:
    """把展品 ID 清单解析为 (请求 ID, 规范档案 ID)；不存在 → 404，解析后重复 → 422。"""
    resolved: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for eid in exhibit_ids:
        canonical = resolve_canonical(conn, eid)
        if canonical is None:
            raise HTTPException(status_code=404, detail=f"展品不存在: {eid}")
        if canonical in seen:
            raise HTTPException(
                status_code=422,
                detail=f"展品 {eid}（规范档案 {canonical}）在清单中重复："
                       "别名解析后同一规范档案出现多次",
            )
        seen[canonical] = eid
        resolved.append((eid, canonical))
    return resolved


def _load_round_result(conn: sqlite3.Connection, round_id: str) -> dict:
    row = conn.execute(
        "SELECT result_json FROM inventory_rounds WHERE round_id = ?", (round_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"盘点轮次不存在或尚未对账: {round_id}"
        )
    return json.loads(row["result_json"])


# ---------- 核验与固化 ----------

def _verify_all(
    conn: sqlite3.Connection,
    payload: OpeningBatchVerifyRequest,
    resolved: list[tuple[str, str]],
    round_result: dict,
    start_s: float,
    end_s: float,
) -> tuple[list[dict], list[dict]]:
    """逐件计算证据与不合规原因；返回 (逐件结果, 参与证据摘要的规范项)。"""
    exhibits_out: list[dict] = []
    digest_items: list[dict] = []
    for requested_id, canonical in resolved:
        inventory = _inventory_evidence(round_result, canonical)
        evidence = _compute_exhibit_evidence(conn, canonical, start_s, end_s, payload)
        reasons = _failure_reasons(payload.round_id, inventory, evidence)
        evidence_out = {"inventory": inventory, **evidence}
        exhibits_out.append({
            "requested_id": requested_id,
            "canonical_id": canonical,
            "compliant": not reasons,
            "failure_reasons": reasons,
            "evidence": evidence_out,
        })
        digest_items.append({"canonical_id": canonical, "evidence": evidence_out})
    return exhibits_out, digest_items


def verify_opening_batch(
    conn: sqlite3.Connection, payload: OpeningBatchVerifyRequest
) -> OpeningBatchVerifyOut:
    """执行一批展品的开放合规核验；全部通过才固化凭证，任一失败不写入。"""
    start_s = ensure_aware(payload.start).timestamp()
    end_s = ensure_aware(payload.end).timestamp()
    if end_s <= start_s:
        raise HTTPException(status_code=422, detail="end 必须晚于 start")

    # 幂等回放不依赖写锁之外的状态：先读一次（无行则进入写事务）。
    existing = conn.execute(
        "SELECT * FROM opening_batch_credentials WHERE request_key = ?",
        (payload.request_key,),
    ).fetchone()
    if existing is not None:
        return _replay(conn, existing, payload, start_s, end_s)

    conn.execute("BEGIN IMMEDIATE")
    try:
        # 锁内重读：并发的相同 request_key 可能已完成落库
        existing = conn.execute(
            "SELECT * FROM opening_batch_credentials WHERE request_key = ?",
            (payload.request_key,),
        ).fetchone()
        if existing is not None:
            conn.commit()  # 尚未写入，COMMIT 仅释放 IMMEDIATE 锁
            return _replay(conn, existing, payload, start_s, end_s)

        # 别名解析、轮次固化结果读取、证据重算全部在同一个写事务内完成
        resolved = _resolve_exhibits(conn, list(payload.exhibit_ids))
        round_result = _load_round_result(conn, payload.round_id)
        canonical_ids = [canonical for _, canonical in resolved]
        request_digest = _request_digest(
            payload.round_id, start_s, end_s, payload, canonical_ids
        )
        exhibits_out, digest_items = _verify_all(
            conn, payload, resolved, round_result, start_s, end_s
        )
        evidence_digest = _evidence_digest(digest_items)

        base = {
            "request_key": payload.request_key,
            "round_id": payload.round_id,
            "start": _to_iso(start_s),
            "end": _to_iso(end_s),
            "max_rate_lux_per_second": payload.max_rate_lux_per_second,
            "stuck_tolerance_lux": payload.stuck_tolerance_lux,
            "min_stuck_seconds": payload.min_stuck_seconds,
            "request_digest": request_digest,
            "evidence_digest": evidence_digest,
            "exhibits": exhibits_out,
        }

        if any(not x["compliant"] for x in exhibits_out):
            # 任一失败：返回逐件原因且不写入（request_key 不被占用）
            conn.rollback()
            return OpeningBatchVerifyOut(
                **base,
                compliant=False,
                credential_id=None,
                digest=None,
                created_at=None,
                already_committed=False,
            )

        # 全部通过：固化包含各项证据摘要的合规凭证
        credential_id = uuid.uuid4().hex
        digest = content_hash_of({
            "request_digest": request_digest,
            "evidence_digest": evidence_digest,
        })
        created_at = _utc_now_iso()
        credential = {
            "credential_id": credential_id,
            **base,
            "digest": digest,
            "created_at": created_at,
        }
        conn.execute(
            """INSERT INTO opening_batch_credentials
               (request_key, credential_id, round_id, start_s, end_s,
                max_rate_lux_per_second, stuck_tolerance_lux, min_stuck_seconds,
                request_digest, evidence_digest, digest, credential_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                payload.request_key, credential_id, payload.round_id, start_s, end_s,
                payload.max_rate_lux_per_second, payload.stuck_tolerance_lux,
                payload.min_stuck_seconds, request_digest, evidence_digest, digest,
                json.dumps(credential, ensure_ascii=False, sort_keys=True),
                created_at,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return OpeningBatchVerifyOut(
        **{k: v for k, v in credential.items() if k != "credential_id"},
        compliant=True,
        credential_id=credential_id,
        already_committed=False,
    )


def _replay(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    payload: OpeningBatchVerifyRequest,
    start_s: float,
    end_s: float,
) -> OpeningBatchVerifyOut:
    """相同 request_key 重放：规范化输入与当时证据摘要都一致才回放，否则 409。"""
    resolved = _resolve_exhibits(conn, list(payload.exhibit_ids))
    canonical_ids = [canonical for _, canonical in resolved]
    request_digest = _request_digest(
        payload.round_id, start_s, end_s, payload, canonical_ids
    )
    if request_digest != row["request_digest"]:
        raise HTTPException(
            status_code=409,
            detail=f"request_key {payload.request_key} 已绑定另一份核验请求"
                   "（盘点轮次/时间窗/展品清单/诊断阈值不同），不能换目标重复提交",
        )
    # 当时摘要核对：按当前库内状态重算逐件证据，任一相关状态变化即不一致。
    # 轮次结果固化后不可变；日历删除等导致无法重算时同样视为不一致。
    try:
        round_result = _load_round_result(conn, row["round_id"])
        _exhibits, digest_items = _verify_all(
            conn, payload, resolved, round_result, start_s, end_s
        )
        current_evidence_digest = _evidence_digest(digest_items)
    except HTTPException:
        raise HTTPException(
            status_code=409,
            detail=f"request_key {payload.request_key} 核验时的状态已无法重算"
                   "（盘点轮次或规范档案不存在），不能重放首次凭证",
        )
    if current_evidence_digest != row["evidence_digest"]:
        raise HTTPException(
            status_code=409,
            detail=f"request_key {payload.request_key} 核验时的证据摘要已变化"
                   "（读数/作废/来源优先级/日历/处置单状态或归并归属已变），"
                   "不能重放首次凭证；请换用新的 request_key 重新核验",
        )
    credential = json.loads(row["credential_json"])
    return OpeningBatchVerifyOut(
        **{k: v for k, v in credential.items() if k != "credential_id"},
        compliant=True,
        credential_id=credential["credential_id"],
        already_committed=True,
    )


# ---------- 查询 ----------

def get_credential(
    conn: sqlite3.Connection, credential_id: str
) -> OpeningBatchCredentialOut | None:
    """按 credential_id 读取已固化的合规凭证；不存在返回 None。"""
    row = conn.execute(
        "SELECT credential_json FROM opening_batch_credentials WHERE credential_id = ?",
        (credential_id,),
    ).fetchone()
    if row is None:
        return None
    return OpeningBatchCredentialOut(**json.loads(row["credential_json"]))
