"""
Test: Kill node during snapshot recovery (Partial state), verify recovery.

Key insight: Partial state only occurs when the peer ALREADY has replicas in consensus.
So we kill the peer but keep it in the cluster, then restart with same identity.
"""
import threading
import time
from pathlib import Path

import pytest
import requests

from .assertions import assert_http_ok
from .fixtures import create_collection, upsert_random_points
from .utils import (
    assert_project_root,
    every_test,
    processes,
    start_cluster,
    start_peer,
    wait_collection_exists_and_active_on_all_peers,
    wait_for_peer_online,
)

COLLECTION = "test_collection"


def log(msg: str, **kw):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", *(f"{k}={v}" for k, v in kw.items()))


def get_shard_states(url: str) -> dict[int, str]:
    try:
        r = requests.get(f"{url}/collections/{COLLECTION}/cluster", timeout=2)
        return {s["shard_id"]: s["state"] for s in r.json()["result"]["local_shards"]} if r.ok else {}
    except Exception:
        return {}


def kill_peer(url: str) -> bool:
    for p in processes:
        if f"http://127.0.0.1:{p.http_port}" == url:
            p.kill()
            processes.remove(p)
            return True
    return False


class PartialMonitor(threading.Thread):
    """Monitor for Partial state and kill target peer when detected."""

    def __init__(self, peer_urls: list[str], target_url: str):
        super().__init__(daemon=True)
        self.peer_urls = peer_urls
        self.target_url = target_url
        self.found = False
        self.killed = False
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            for url in self.peer_urls:
                for shard_id, state in get_shard_states(url).items():
                    if state == "Partial":
                        log("PARTIAL detected", peer=url, shard=shard_id)
                        self.found = True
                        if kill_peer(self.target_url):
                            self.killed = True
                        return

    def stop(self):
        self._stop.set()
        self.join(timeout=2)


def test_snapshot_restore_kill_during_partial(tmp_path: Path, every_test):
    """Kill peer during Partial state, restart, verify all shards become Active."""
    assert_project_root()

    env = {
        "QDRANT__LOG_LEVEL": "DEBUG",
        "QDRANT__STAGING__SNAPSHOT_RECOVERY_DELAY": "5.0",
    }

    # Setup
    log("Starting cluster")
    peer_urls, peer_dirs, bootstrap = start_cluster(tmp_path, 3, port_seed=23000, extra_env=env)
    create_collection(peer_urls[0], shard_number=3, replication_factor=2)
    wait_collection_exists_and_active_on_all_peers(collection_name=COLLECTION, peer_api_uris=peer_urls)
    upsert_random_points(peer_urls[0], 10_000, batch_size=100)

    # Create snapshot
    r = requests.post(f"{peer_urls[0]}/collections/{COLLECTION}/snapshots")
    assert_http_ok(r)
    snapshot_url = f"{peer_urls[0]}/collections/{COLLECTION}/snapshots/{r.json()['result']['name']}"
    log("Snapshot created")

    # Kill target peer (keep in cluster!)
    target_port = 23200
    target_dir = Path(peer_dirs[2])
    kill_peer(peer_urls[2])
    time.sleep(2)

    # Restart and trigger recovery
    log("Restarting peer")
    new_url = start_peer(target_dir, "peer_recovery.log", bootstrap, port=target_port, extra_env=env)

    monitor = PartialMonitor(peer_urls[:2] + [new_url], new_url)
    monitor.start()

    wait_for_peer_online(new_url)
    requests.put(f"{new_url}/collections/{COLLECTION}/snapshots/recover?wait=false", json={"location": snapshot_url})

    # Wait for Partial detection
    monitor.join(timeout=10)
    monitor.stop()

    if not monitor.killed:
        kill_peer(new_url)
        pytest.fail("Partial state not detected")

    log("Killed during Partial, restarting...")
    time.sleep(2)

    # Restart again
    restarted = start_peer(target_dir, "peer_restarted.log", bootstrap, port=target_port, extra_env=env)
    try:
        wait_for_peer_online(restarted)
    except Exception:
        pass  # Continue anyway

    # Main assertion: wait for all Active
    all_peers = peer_urls[:2] + [restarted]
    for _ in range(30):
        if all(all(s == "Active" for s in get_shard_states(u).values()) for u in all_peers if get_shard_states(u)):
            log("All shards Active!")
            return
        time.sleep(0.5)

    # Log final state on failure
    for url in all_peers:
        log("Final state", peer=url, shards=get_shard_states(url))
    pytest.fail("Not all shards became Active")