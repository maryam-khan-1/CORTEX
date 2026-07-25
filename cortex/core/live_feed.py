"""Live threat-feed simulator for the SOC dashboard UI.

Uses fast keyword heuristics so the stream stays real-time; optional
model triage can be triggered in batches without blocking the feed.
"""

from __future__ import annotations

import random
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Optional

# Expanding corpus — multilingual, SOC-flavored
FEED_CORPUS: list[dict[str, str]] = [
    {"src": "auth", "msg": "Failed login for user admin from 203.0.113.50 — 12 attempts in 30s"},
    {"src": "fw", "msg": "ALLOW tcp 22 from 198.51.100.9 to bastion-01"},
    {"src": "ids", "msg": "检测到 PowerShell 下载 cradle: IEX (New-Object Net.WebClient).DownloadString"},
    {"src": "web", "msg": "Error 500 en /api/users?id=1' OR '1'='1 — posible inyección SQL"},
    {"src": "dns", "msg": "Query for update.microsoft.com from workstation-22 NXDOMAIN then success"},
    {"src": "edr", "msg": "Process created: mimikatz.exe parent=winword.exe — credential dumping suspected"},
    {"src": "proxy", "msg": "正常浏览 https://example.com/docs/readme"},
    {"src": "vpn", "msg": "Usuario maria conectada desde oficina — handshake OK"},
    {"src": "k8s", "msg": "Unauthorized access to secrets API from pod crypto-miner-7f9b"},
    {"src": "app", "msg": "INFO healthcheck ok latency=12ms"},
    {"src": "email", "msg": "Phishing lure: invoice.pdf.js attached — macro execution blocked"},
    {"src": "cloud", "msg": "IAM role AssumeRole from anomalous geo: 102.0.0.44"},
    {"src": "edr", "msg": "LSASS memory access by unexpected process rundll32.exe"},
    {"src": "fw", "msg": "DENY outbound 4444/tcp to 185.220.101.23 (Tor exit)"},
    {"src": "web", "msg": "Path traversal attempt /../../etc/passwd on edge-proxy"},
    {"src": "dns", "msg": "长域名隧道可疑: a1b2c3.exfil.evil.example 高频查询"},
    {"src": "auth", "msg": "MFA fatigue: 19 push denies then one accept for svc-backup"},
    {"src": "k8s", "msg": "Privileged container started without admission review"},
    {"src": "app", "msg": "Scheduled job completed ok rows=412"},
    {"src": "ids", "msg": "Cobalt Strike beacon JA3 match on host finance-laptop"},
    {"src": "vpn", "msg": "Sesión VPN renovada — sin anomalías"},
    {"src": "web", "msg": "WAF blocked XSS payload <script>alert(1)</script> on /search"},
    {"src": "cloud", "msg": "S3 public ACL change on bucket customer-exports"},
    {"src": "edr", "msg": "Suspicious scheduled task: \\Microsoft\\Windows\\UpdateOrchestrator\\Backup"},
]

CRITICAL_KEYS = (
    "mimikatz", "lsass", "cobalt", "crypto-miner", "exfil", "ransom",
    "credential dump", "beacon",
)
SUSPICIOUS_KEYS = (
    "failed login", "powershell", "sql", "inyección", "phishing", "assumerole",
    "path traversal", "mfa fatigue", "privileged", "waf blocked", "xss",
    "public acl", "deny outbound", "tor", "unauthorized", "下载", "隧道",
)
MITRE_MAP = {
    "mimikatz": "T1003",
    "lsass": "T1003",
    "powershell": "T1059.001",
    "sql": "T1190",
    "inyección": "T1190",
    "phishing": "T1566",
    "beacon": "T1071",
    "exfil": "T1048",
    "crypto-miner": "T1496",
    "mfa": "T1621",
    "tor": "T1090",
    "xss": "T1059",
    "acl": "T1485",
}


@dataclass
class FeedEvent:
    ts: str
    source: str
    message: str
    label: str  # benign | suspicious | critical
    mitre: str
    consensus: str
    score: float  # 0-1 heuristic confidence stand-in for display

    def as_row(self) -> list[Any]:
        return [self.ts, self.source, self.label, self.mitre, self.consensus, self.message[:140]]


def classify_heuristic(message: str) -> tuple[str, str, float]:
    low = message.lower()
    for k in CRITICAL_KEYS:
        if k in low:
            mitre = next((v for key, v in MITRE_MAP.items() if key in low), "T1059")
            return "critical", mitre, 0.92
    for k in SUSPICIOUS_KEYS:
        if k in low:
            mitre = next((v for key, v in MITRE_MAP.items() if key in low), "T1078")
            return "suspicious", mitre, 0.78
    return "benign", "—", 0.88


@dataclass
class LiveFeedState:
    events: Deque[FeedEvent] = field(default_factory=lambda: deque(maxlen=80))
    running: bool = True
    tick: int = 0
    critical_total: int = 0
    suspicious_total: int = 0
    benign_total: int = 0
    history: list[dict[str, int]] = field(default_factory=list)  # time-series counts

    def push_random(self) -> FeedEvent:
        item = random.choice(FEED_CORPUS)
        label, mitre, score = classify_heuristic(item["msg"])
        # light jitter so consensus % looks alive
        agree = 3 if score > 0.85 else (2 if score > 0.7 else 1)
        consensus = f"{label.capitalize()} ({agree}/3 agree)"
        ev = FeedEvent(
            ts=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            source=item["src"],
            message=item["msg"],
            label=label,
            mitre=mitre,
            consensus=consensus,
            score=score,
        )
        self.events.appendleft(ev)
        self.tick += 1
        if label == "critical":
            self.critical_total += 1
        elif label == "suspicious":
            self.suspicious_total += 1
        else:
            self.benign_total += 1
        # sample every event into sparkline history (keep last 40)
        self.history.append(
            {
                "t": self.tick,
                "critical": self.critical_total,
                "suspicious": self.suspicious_total,
                "benign": self.benign_total,
            }
        )
        if len(self.history) > 40:
            self.history = self.history[-40:]
        return ev

    def seed(self, n: int = 12) -> None:
        for _ in range(n):
            self.push_random()
            time.sleep(0)  # yield

    def counts(self) -> Counter:
        return Counter(e.label for e in self.events)

    def table_rows(self) -> list[list[Any]]:
        return [e.as_row() for e in list(self.events)[:40]]

    def snapshot(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "critical_total": self.critical_total,
            "suspicious_total": self.suspicious_total,
            "benign_total": self.benign_total,
            "window": dict(self.counts()),
            "events": [asdict(e) for e in list(self.events)[:25]],
        }


FEED = LiveFeedState()
FEED.seed(14)
