from stateaware.config import Settings
from stateaware.judge import HarmBenchJudge


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self):
        self.calls = []

    def post(self, url, json):
        self.calls.append((url, json))
        if url.endswith("/health"):
            return FakeResponse({"status": "ok"})
        if url.endswith("/score_full"):
            return FakeResponse({
                "asr": 1,
                "probability": 0.999,
                "logit_yes": 13.3,
                "logit_no": 3.75,
                "logit_diff": 9.55,
                "version": "test-full",
            })
        return FakeResponse({"asr": 0, "probability": 0.2, "version": "test-score"})

    def close(self):
        return None


def make_settings() -> Settings:
    return Settings(
        target_base_url="http://10.50.2.40:11434/v1",
        target_api_key="ollama",
        target_model="llama3.1:8b",
        target_temperature=0.0,
        target_max_tokens=64,
        target_timeout_seconds=10.0,
        target_seed=17,
        judge_score_url="http://judge/score",
        judge_score_full_url="http://judge/score_full",
        judge_health_url="http://judge/health",
        judge_timeout_seconds=10.0,
    )


def test_health_uses_post_with_empty_json(monkeypatch):
    judge = HarmBenchJudge(make_settings())
    fake = FakeHttpClient()
    monkeypatch.setattr(judge, "client", fake)
    assert judge.health() == {"status": "ok"}
    assert fake.calls == [("http://judge/health", {})]


def test_score_routes_and_preserves_full_logits(monkeypatch):
    judge = HarmBenchJudge(make_settings())
    fake = FakeHttpClient()
    monkeypatch.setattr(judge, "client", fake)

    canonical = judge.score("behavior", "generation", mode="score")
    full = judge.score("behavior", "generation", mode="full")

    assert canonical.asr == 0
    assert canonical.endpoint_mode == "score"
    assert full.asr == 1
    assert full.logit_diff == 9.55
    assert fake.calls[0] == (
        "http://judge/score", {"behavior": "behavior", "generation": "generation"}
    )
    assert fake.calls[1] == (
        "http://judge/score_full", {"behavior": "behavior", "generation": "generation"}
    )
