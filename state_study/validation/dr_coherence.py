"""Is the delayed-relevance L2 correction coherent with the history it corrects?

Runs its Warehouse under the oracle policy and, for the shelf the correction_notice empties,
compares the put-away the notice names with the put-away that actually holds the shelf when the
notice arrives.   python dr_coherence.py [--seeds 40] [--src PATH]"""
import argparse
import re

from dr_common import import_dr


def check(Warehouse, seed, k=10, horizon=50):
    env = Warehouse(horizon=horizon, seed=seed, invalidation_k=k)
    env.reset()
    origin, shipped = {}, set()          # shelf -> step of its current put-away; shipped put-aways
    t_c, s_c = env.invalidation_from, env.invalidated_shelf
    while not env.done:
        obs = env.observe()
        if obs.step == t_c:
            named = int(re.search(r"corrects_step=(\d+)", obs.text).group(1))
            occ = origin.get(s_c)
            return dict(seed=seed, notice_step=t_c, shelf=s_c, names_step=named, occupant_step=occ,
                        named_pallet_shipped=named in shipped, coherent=(occ == named))
        a = env.expected_action()
        if a.name == "Store":
            origin[a.args["shelf"]] = obs.step
        elif a.name == "Ship":
            shipped.add(origin.pop(a.args["shelf"], None))
        env.apply(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--src", default=None)
    a = p.parse_args()
    Warehouse, _ = import_dr(a.src)
    rows = []
    for s in range(a.seeds):
        try:
            rows.append(check(Warehouse, s))
        except ValueError:
            pass
    for r in rows:
        if r["seed"] in (0, 1, 2, 4, 6, 10):
            print("seed used in its L2 runs:", r)
    print(f"coherent notices over seeds 0..{a.seeds - 1}: {sum(r['coherent'] for r in rows)}/{len(rows)}")
