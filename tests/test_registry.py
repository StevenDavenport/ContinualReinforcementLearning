from crlbench.core.registry import Registry, RegistryError


def test_registry_register_and_create() -> None:
    registry: Registry[int] = Registry(name="test")
    registry.register("one", lambda: 1)
    assert registry.create("one") == 1
    assert registry.names() == ("one",)


def test_registry_rejects_duplicate_without_replace() -> None:
    registry: Registry[int] = Registry(name="test")
    registry.register("value", lambda: 1)
    try:
        registry.register("value", lambda: 2)
        raise AssertionError("Expected duplicate registration to fail.")
    except RegistryError:
        pass


def test_registry_missing_key_raises() -> None:
    registry: Registry[int] = Registry(name="test")
    try:
        registry.create("missing")
        raise AssertionError("Expected missing key lookup to fail.")
    except RegistryError:
        pass
