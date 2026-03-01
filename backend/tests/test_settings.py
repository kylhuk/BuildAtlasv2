from backend.app.settings import Settings


def test_pob_worker_pool_size_defaults_to_double_cpu(monkeypatch):
    monkeypatch.setattr("backend.app.settings.os.cpu_count", lambda: 12)
    monkeypatch.delenv("POB_WORKER_POOL_SIZE", raising=False)
    settings = Settings(_env_file=None)
    assert settings.pob_worker_pool_size == 24


def test_pob_worker_pool_size_uses_explicit_environment(monkeypatch):
    monkeypatch.setattr("backend.app.settings.os.cpu_count", lambda: 12)
    monkeypatch.setenv("POB_WORKER_POOL_SIZE", "6")
    settings = Settings(_env_file=None)
    assert settings.pob_worker_pool_size == 6


def test_pob_worker_pool_size_prefers_env_over_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "backend.env"
    env_file.write_text("POB_WORKER_POOL_SIZE=3\n")

    monkeypatch.setattr("backend.app.settings.os.cpu_count", lambda: 12)
    monkeypatch.setenv("POB_WORKER_POOL_SIZE", "7")
    settings_with_env = Settings(_env_file=str(env_file))
    assert settings_with_env.pob_worker_pool_size == 7

    monkeypatch.delenv("POB_WORKER_POOL_SIZE")
    settings_with_file = Settings(_env_file=str(env_file))
    assert settings_with_file.pob_worker_pool_size == 3
