"""Per-step ground-truth world state for ALFWorld (proposal I2, Sec. 3.6).

ALFWorld builds its TextWorld environments without ``facts`` by default, so PoS's adapter
never sees the true world state. This module opens the same valid-unseen games with
``EnvInfos(facts=True)`` and returns, after every action, the task-relevant facts:

    agent location, held object, object -> receptacle, opened receptacles,
    and object properties (clean / hot / cool / sliced / toggled).

Run as a script to replay the handcoded expert on a few games and check that the state
changes the way each action says it should -- no language model involved:

    python -m state_study.groundtruth.alfworld_facts --data DIR --games 5
"""

from __future__ import annotations

import argparse
import glob
import os
import re

TRACKED = {"atlocation", "holds", "inreceptacle", "opened", "isclean", "ishot", "iscool",
           "issliced", "istoggled"}


def make_env(game_file: str, max_steps: int = 80):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler, AlfredExpert, AlfredExpertType, AlfredInfos)

    infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True,
                               extras=["gamefile", "expert_plan"])
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert(AlfredExpertType.HANDCODED)]
    env_id = textworld.gym.register_games([game_file], infos, batch_size=1, asynchronous=False,
                                          max_episode_steps=max_steps, wrappers=wrappers)
    return textworld.gym.make(env_id)


def _name(var) -> str:
    return getattr(var, "name", str(var))


def state_from_facts(facts) -> dict:
    """Project TextWorld propositions onto the fields a state schema would track."""
    st = {"agent_at": None, "holding": [], "in": {}, "opened": [], "props": {}}
    for p in facts:
        pred = p.name.lower()
        if pred not in TRACKED:
            continue
        args = [_name(a) for a in p.arguments]
        if pred == "atlocation" and args and args[0].lower().startswith("agent"):
            st["agent_at"] = args[-1]
        elif pred == "holds":
            st["holding"].append(args[-1])
        elif pred == "inreceptacle":
            st["in"][args[0]] = args[1]
        elif pred == "opened":
            st["opened"].append(args[0])
        elif pred.startswith("is"):
            st["props"].setdefault(args[0], []).append(pred[2:])
    st["holding"].sort(); st["opened"].sort()
    return st


def replay_expert(game_file: str, max_steps: int = 80) -> list[dict]:
    """Follow the handcoded expert; return one record per step with action and true state."""
    env = make_env(game_file, max_steps)
    obs, infos = env.reset()
    records = [{"step": 0, "action": None, "state": state_from_facts(infos["facts"][0])}]
    for t in range(1, max_steps + 1):
        plan = infos["extra.expert_plan"][0]
        if not plan:
            break
        action = plan[0]
        obs, scores, dones, infos = env.step([action])
        records.append({"step": t, "action": action, "state": state_from_facts(infos["facts"][0]),
                        "won": bool(infos["won"][0])})
        if dones[0]:
            break
    env.close()
    return records


def check_transitions(records: list[dict]) -> list[str]:
    """Return violations of what each action should do to the true state."""
    problems = []
    for prev, cur in zip(records, records[1:]):
        a, s0, s1 = cur["action"], prev["state"], cur["state"]
        if m := re.match(r"take (.+) from (.+)", a):
            obj = m.group(1)
            if not any(h.startswith(obj.split()[0]) for h in s1["holding"]):
                problems.append(f"step {cur['step']}: '{a}' but holding={s1['holding']}")
        elif m := re.match(r"(?:move|put) (.+?) (?:to|in/on) (.+)", a):
            if s1["holding"] and s1["holding"] == s0["holding"]:
                problems.append(f"step {cur['step']}: '{a}' but still holding {s1['holding']}")
        elif a.startswith("go to "):
            if s1["agent_at"] == s0["agent_at"] and s0["agent_at"] is not None:
                problems.append(f"step {cur['step']}: '{a}' but agent stayed at {s1['agent_at']}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing json_2.1.1/valid_unseen")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1, help="take every k-th game (spread task types)")
    args = ap.parse_args()
    os.environ.setdefault("ALFWORLD_DATA", args.data)
    games = sorted(glob.glob(os.path.join(args.data, "json_2.1.1/valid_unseen/*/*/game.tw-pddl")))
    print(f"{len(games)} valid-unseen games found")
    for gf in games[:: args.stride][: args.games]:
        recs = replay_expert(gf)
        probs = check_transitions(recs)
        changed = sum(r["state"] != p["state"] for p, r in zip(recs, recs[1:]))
        task = gf.split("valid_unseen/")[1].split("/")[0]
        print(f"{task[:55]:55s} steps={len(recs) - 1:3d} won={recs[-1].get('won')} "
              f"state_changed_steps={changed:3d} objects_tracked={len(recs[0]['state']['in']):3d} "
              f"violations={len(probs)}")
        for p in probs[:3]:
            print("   ", p)


if __name__ == "__main__":
    main()
