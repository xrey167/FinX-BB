from so.experiments.e000051b_clean_bystanders import clean_other_pod_aliases


def test_clean_other_pod_aliases_excludes_target_pod():
    groups = [((i, 0), [(i, 1), (i, 2)]) for i in range(100)]
    for target, aliases in groups:
        picked = clean_other_pod_aliases(groups, target, 2)
        assert len(picked) == 4
        assert set(picked).isdisjoint({target, *aliases})


def test_original_fixed_tail_selection_can_contaminate():
    groups = [((i, 0), [(i, 1), (i, 2)]) for i in range(100)]
    fixed = [k for _, ks in groups[-2:] for k in ks]
    contaminated = 0
    for target, aliases in groups:
        if set(fixed).intersection({target, *aliases}):
            contaminated += 1
    assert contaminated == 2
