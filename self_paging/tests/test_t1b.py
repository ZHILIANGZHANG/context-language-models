from self_paging.envs.t1b_lookup import LookupEnv, generate, rule_lookup


def test_lookup_exact_sufficiency_and_no_current_page_shortcuts():
    for seed in range(20):
        ep = generate(seed)
        assert ep == generate(seed)
        assert len(ep.decisions) == 4
        for d in ep.decisions:
            assert d.evidence_page < d.page
            assert rule_lookup(ep.pages[:d.page+1], d.entity) == d.answer
            assert rule_lookup([ep.pages[i] for i in sorted(ep.need[d.page])], d.entity) == d.answer
            assert rule_lookup([ep.pages[i] for i in range(d.page+1) if i!=d.evidence_page],d.entity) is None
            assert d.answer not in ep.pages[d.page]
            assert all(d.entity not in p for p in ep.pages[:d.evidence_page])


def test_lookup_observations_independent_of_actions_and_exact_scoring():
    good,bad=LookupEnv(0),LookupEnv(0)
    while not good.done:
        if good.pending is not None:
            value=good.pending.answer
            good.step('lookup',{'value':value})
            bad.step('lookup',{'value':'wrong'})
        else:
            assert good.step('next_records',{}) == bad.step('next_records',{})
    assert good.score()==1. and bad.score()==0.
