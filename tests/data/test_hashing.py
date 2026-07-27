from fl_brats.data.hashing import sha256_payload


def test_payload_hash_is_order_independent() -> None:
    assert sha256_payload({"a": 1, "b": 2}) == sha256_payload({"b": 2, "a": 1})
