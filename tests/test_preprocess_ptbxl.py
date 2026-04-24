import numpy as np
import pandas as pd

from scripts.preprocess_ptbxl import (
    build_diagnostic_mapping,
    build_splits,
    choose_superclass_label,
    parse_scp_codes,
)


def test_parse_scp_codes_valid_dict():
    parsed = parse_scp_codes("{'AMI': 80, 'IMI': 50}")
    assert parsed == {'AMI': 80, 'IMI': 50}


def test_parse_scp_codes_non_dict_returns_empty():
    parsed = parse_scp_codes("['AMI', 'IMI']")
    assert parsed == {}


def test_choose_superclass_label_strict_accepts_consistent_mapping():
    mapping = {'AMI': 'MI', 'IMI': 'MI'}
    label = choose_superclass_label("{'AMI': 80, 'IMI': 50}", mapping, strict_single_superclass=True)
    assert label == 'MI'


def test_choose_superclass_label_strict_rejects_conflicting_mapping():
    mapping = {'AMI': 'MI', 'LVH': 'HYP'}
    label = choose_superclass_label("{'AMI': 80, 'LVH': 50}", mapping, strict_single_superclass=True)
    assert label is None


def test_choose_superclass_label_non_strict_uses_highest_likelihood_code():
    mapping = {'AMI': 'MI', 'LVH': 'HYP'}
    label = choose_superclass_label("{'AMI': 40, 'LVH': 90}", mapping, strict_single_superclass=False)
    assert label == 'HYP'


def test_build_diagnostic_mapping_filters_non_diagnostic_and_missing_superclass():
    scp_df = pd.DataFrame(
        {
            'diagnostic': [1, 1, 0, 1],
            'diagnostic_class': ['MI', np.nan, 'IGNORED', 'HYP'],
        },
        index=['AMI', 'MISC', 'NOPE', 'LVH'],
    )

    mapping = build_diagnostic_mapping(scp_df)
    assert mapping == {'AMI': 'MI', 'LVH': 'HYP'}


def test_build_splits_uses_ptbxl_fold_convention():
    df = pd.DataFrame({'strat_fold': [1, 8, 9, 10]})
    X = np.arange(4 * 2 * 3).reshape(4, 2, 3)
    y = np.array([0, 1, 2, 3])

    X_train, y_train, X_val, y_val, X_test, y_test = build_splits(df, X, y)

    assert X_train.shape[0] == 2
    assert y_train.tolist() == [0, 1]
    assert X_val.shape[0] == 1
    assert y_val.tolist() == [2]
    assert X_test.shape[0] == 1
    assert y_test.tolist() == [3]
