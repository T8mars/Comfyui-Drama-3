"""Protect exact seed values crossing a browser's JSON number boundary."""
import pytest

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.nodes import sampling_seed


@pytest.mark.parametrize('value', [0, 42, 2**53-1, 2**53+1, 2**64-1])
def test_decimal_seed_override_preserves_every_digit(value):
    assert sampling_seed(42, str(value)) == value
    assert sampling_seed(42, '  '+str(value)+'  ') == value
    assert type(sampling_seed(42, str(value))) is int


@pytest.mark.parametrize('value', ['-1', '1.0', '1e3', '18446744073709551616',
                                 '123456789012345678901', '１２３', True, None])
def test_invalid_exact_seed_is_rejected_before_inference(value):
    with pytest.raises(ValueError):
        sampling_seed(42, value)


@pytest.mark.parametrize('value', [True, 1.5, -1, 2**64])
def test_invalid_numeric_seed_is_not_silently_coerced(value):
    with pytest.raises(ValueError):
        sampling_seed(value)


def test_blank_override_keeps_existing_exact_python_seed_behavior():
    assert sampling_seed(2**64-1) == 2**64-1
    assert sampling_seed(42, ' ') == 42
