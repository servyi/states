#!/usr/bin/env python3
"""
Explicit-state explorer for the checkpointer park-handshake protocol.
Mirrors src/checkpoint.rs semantics atomically:

Shared fields per pair: pending, closed, parked (bools).
Condvar waits modeled as blocking until the recheck-condition holds
(Mesa semantics: wait = release lock, block, relock, recheck).

Roles:
  MAIN  : top handle. serve loop { if pending_top: park_self; if all leaves
          closed: DONE; (sleep) } ; park_self = retain(map child:
          request_until_parked(None)) then park self, wait pending_top==0,
          unpark, then release every retained child.
  TIMER : loop { request_timeout(50ms): pending_top=1; block until
          parked_top==1 (-> GUARD: hold guard, then drop ->
          release_request(top): pending_top=0, wait parked_top==0)
          OR timeout (nondet while waiting): pending_top=0, RETRY }.
  LEAF i: loop { safepoint: if pending_i: park (parked_i=1; wait pending_i==0;
          parked_i=0); work } ; finish: closed_i=1 (handle drop; notifies).

Deadlock of interest: MAIN blocked in request_until_parked(c) with
pending_c==0 && closed_c==0 && leaf not going to park (pending stays 0),
i.e. the wait can never be enabled again. We detect reachable states where
MAIN is blocked and its blocking condition is permanently disabled.

Interleaving semantics: at each step pick any thread that has an enabled
atomic action. Timeouts/finish/work-length are nondeterministic.
"""
import itertools, sys
from collections import deque

N = 4  # leaves

# ---- thread program counters ----
# MAIN: 'serve', 'req_child:<idx-set progress>', 'parked_top', 'releasing:<idx>', 'done'
# TIMER: 'idle_start', 'req_wait', 'guard', 'rel_wait', 'retry_sleep'
# LEAF i: 'run', 'park_wait', 'closed'

# State = (main_pc, main_req_idx, main_rel_idx, timer_pc,
#          tuple of leaf pcs (4),
#          pending_top, parked_top,
#          tuple pending_i, tuple parked_i, tuple closed_i,
#          children_kept bitmask)

def explore():
    start = (
        'serve',            # main_pc
        0,                  # main_req_idx (index into children list during retain)
        0,                  # main_rel_idx (release loop index)
        'idle_start',       # timer_pc
        tuple(['run']*N),   # leaf pcs
        0, 0,               # pending_top, parked_top
        tuple([0]*N), tuple([0]*N), tuple([0]*N),
        (1<<N) - 1,         # children_kept bitmask (all registered)
    )
    seen = {start}
    q = deque([start])
    bad = []
    while q:
        s = q.popleft()
        for t in transitions(s):
            if t not in seen:
                seen.add(t)
                q.append(t)
                if is_bad(t):
                    bad.append(t)
                    # keep going a bit to have a witness, then stop early
                    if len(bad) > 0 and len(seen) > 200000:
                        q.clear()
                        break
    return start, seen, bad

def is_bad(s):
    (m, ridx, relidx, tm, lpcs, pt, pkt, pend, park, clo, kept) = s
    # MAIN stuck in request_until_parked(child ridx) where the child will never park:
    # child ridx alive (not closed), pending==0 (flag lost) and leaf is in 'run'
    # (it checks pending only at its safepoint; with pending==0 it never parks).
    if m == 'req_child' and ridx < N:
        i = ridx
        if (kept >> i) & 1 and clo[i] == 0 and pend[i] == 0 and lpcs[i] == 'run':
            return True
    # MAIN stuck waiting for its own resume while no one will resume:
    # parked_top==1, pending_top==0 (timer gave up / lost), timer not active.
    # (Timer retries forever, so this recovers; skip.)
    return False

def transitions(s):
    (m, ridx, relidx, tm, lpcs, pt, pkt, pend, park, clo, kept) = s
    out = []
    L = list(lpcs); P = list(pend); K = list(park); C = list(clo)

    # ---------- LEAF i ----------
    for i in range(N):
        if L[i] == 'run':
            # safepoint check + park entry (atomic under its mutex)
            if P[i] == 1:
                t = list_state(m, ridx, relidx, tm, lpcs, pt, 1, P, K, C, kept)
                t['L'][i] = 'park_wait'; t['K'][i] = 1
                out.append(fin(t))
            else:
                # work step: nondeterministically either keep working or finish
                t = list_state(m, ridx, relidx, tm, lpcs, pt, pkt, P, K, C, kept)
                out.append(fin(t))  # continue working (loop to safepoint)
                t2 = list_state(m, ridx, relidx, tm, lpcs, pt, pkt, P, K, C, kept)
                t2['L'][i] = 'closed'; t2['C'][i] = 1  # action returns, handle drops (close())
                out.append(fin(t2))
        elif L[i] == 'park_wait':
            if P[i] == 0:  # resume: pending cleared by MAIN's release_request
                t = list_state(m, ridx, relidx, tm, lpcs, pt, pkt, P, K, C, kept)
                t['L'][i] = 'run'; t['K'][i] = 0
                out.append(fin(t))
            # else: keep waiting (no transition; condvar blocks)

    # ---------- MAIN ----------
    if m == 'serve':
        if pt == 1:
            # enter park_self: start retain
            t = list_state('req_child', 0, 0, tm, lpcs, pt, pkt, P, K, C, kept)
            out.append(fin(t))
        elif all(c == 1 for c in C):
            t = list_state('done', 0, 0, tm, lpcs, pt, pkt, P, K, C, kept)
            out.append(fin(t))
        else:
            # poll loop iteration (sleep; nothing changes)
            t = list_state(m, ridx, relidx, tm, lpcs, pt, pkt, P, K, C, kept)
            out.append(fin(t))
    elif m == 'req_child':
        if ridx >= N:
            # retain finished -> park self
            t = list_state('parked_top', 0, 0, tm, lpcs, pt, 1, P, K, C, kept)
            out.append(fin(t))
        else:
            i = ridx
            if not ((kept >> i) & 1):
                # already removed in an earlier cycle?? not modeled; skip index
                t = list_state(m, ridx + 1, relidx, tm, lpcs, pt, pkt, P, K, C, kept)
                out.append(fin(t))
            else:
                # request_until_parked(child i): set pending, then block until
                # parked_i==1 (-> Parked) or closed_i==1 (-> Closed, clear pending,
                # remove child).
                if K[i] == 1:
                    t = list_state(m, ridx + 1, relidx, tm, lpcs, pt, pkt, P, K, C, kept)
                    out.append(fin(t))
                elif C[i] == 1:
                    t = list_state(m, ridx + 1, relidx, tm, lpcs, pt, pkt, P, K, C, kept)
                    t['P'][i] = 0; t['kept'] &= ~(1 << i)
                    out.append(fin(t))
                elif P[i] == 0:
                    # pending not yet set: the store is the first statement
                    t = list_state(m, ridx, relidx, tm, lpcs, pt, pkt, P, K, C, kept)
                    t['P'][i] = 1
                    out.append(fin(t))
                # else P[i]==1 and leaf run: MAIN blocks (no transition)
    elif m == 'parked_top':
        if pt == 0:
            # resume: unpark, then release children (in kept order)
            t = list_state('releasing', 0, 0, tm, lpcs, pt, 0, P, K, C, kept)
            out.append(fin(t))
        # else: wait_resume blocks
    elif m == 'releasing':
        if relidx >= N:
            t = list_state('serve', 0, 0, tm, lpcs, pt, pkt, P, K, C, kept)
            out.append(fin(t))
        else:
            i = relidx
            if not ((kept >> i) & 1):
                t = list_state(m, ridx, relidx + 1, tm, lpcs, pt, pkt, P, K, C, kept)
                out.append(fin(t))
            else:
                # release_request(child i): pending=0 under lock, then wait for
                # unpark (parked_i==0 or closed_i==1)
                if P[i] == 1:
                    t = list_state(m, ridx, relidx, tm, lpcs, pt, pkt, P, K, C, kept)
                    t['P'][i] = 0
                    out.append(fin(t))
                elif K[i] == 0 or C[i] == 1:
                    t = list_state(m, ridx, relidx + 1, tm, lpcs, pt, pkt, P, K, C, kept)
                    out.append(fin(t))
                # else: blocked waiting for leaf to observe pending==0 and unpark

    # ---------- TIMER ----------
    if tm == 'idle_start' or tm == 'retry_sleep':
        t = list_state(m, ridx, relidx, 'req_wait', lpcs, 1, pkt, P, K, C, kept)
        out.append(fin(t))  # new request: pending_top=1
    elif tm == 'req_wait':
        if pkt == 1:
            t = list_state(m, ridx, relidx, 'guard', lpcs, pt, pkt, P, K, C, kept)
            out.append(fin(t))
        else:
            # nondeterministic timeout: give up (clear pending) or keep waiting
            t = list_state(m, ridx, relidx, 'retry_sleep', lpcs, 0, pkt, P, K, C, kept)
            out.append(fin(t))
            t2 = list_state(m, ridx, relidx, 'req_wait', lpcs, pt, pkt, P, K, C, kept)
            out.append(fin(t2))
    elif tm == 'guard':
        # drop guard -> release_request(top): pending_top=0 then wait unpark
        t = list_state(m, ridx, relidx, 'rel_wait', lpcs, 0, pkt, P, K, C, kept)
        out.append(fin(t))
    elif tm == 'rel_wait':
        if pkt == 0:
            t = list_state(m, ridx, relidx, 'retry_sleep', lpcs, pt, pkt, P, K, C, kept)
            out.append(fin(t))
        # else: wait for main to unpark (no transition)

    return out

def list_state(m, ridx, relidx, tm, lpcs, pt, pkt, P, K, C, kept):
    return {'m': m, 'ridx': ridx, 'relidx': relidx, 'tm': tm, 'L': list(lpcs),
            'pt': pt, 'pkt': pkt, 'P': list(P), 'K': list(K), 'C': list(C), 'kept': kept}

def fin(t):
    return (t['m'], t['ridx'], t['relidx'], t['tm'], tuple(t['L']),
            t['pt'], t['pkt'], tuple(t['P']), tuple(t['K']), tuple(t['C']), t['kept'])

def fmt(s):
    (m, ridx, relidx, tm, lpcs, pt, pkt, pend, park, clo, kept) = s
    return (f"MAIN={m}(req@{ridx},rel@{relidx}) TIMER={tm} pt={pt} pkt={pkt} "
            f"leaves(l,p,k,c)={list(zip(lpcs,pend,park,clo))} kept={kept:04b}")

if __name__ == '__main__':
    start, seen, bad = explore()
    print(f"states explored: {len(seen)}")
    print(f"BAD states (main stuck on never-parking child): {len(bad)}")
    if bad:
        print("WITNESS:")
        print(" ", fmt(bad[0]))
        # reconstruct a path to it
        # (re-run BFS storing parents)
        parent = {start: None}
        q = deque([start])
        found = None
        while q:
            s = q.popleft()
            if s == bad[0]:
                found = s; break
            for t in transitions(s):
                if t not in parent:
                    parent[t] = s
                    q.append(t)
        if found:
            path = []
            while found:
                path.append(found); found = parent[found]
            print(f"\nTRACE ({len(path)} steps):")
            for i, st in enumerate(reversed(path)):
                print(f"  {i:3d}: {fmt(st)}")
