def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["index_ready"] is True


def test_rag_query(client):
    r = client.post("/api/v1/rag/query", json={"question": "What is RAG?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] and body["citations"]
    assert body.get("retrieval")


def test_agents(client):
    r = client.post("/api/v1/agents/run", json={"task": "Search corpus for evals and calculate 2+2", "approve": True})
    assert r.status_code == 200
    assert r.json()["events"]


def test_evals(client):
    r = client.post("/api/v1/evals/run", json={"limit": 3, "write_report": False})
    assert r.status_code == 200
    assert "pass_rate" in r.json()["summary"]


def test_traces_analytics(client):
    client.post("/api/v1/rag/query", json={"question": "citations", "top_k": 2})
    assert client.get("/api/v1/traces").status_code == 200
    a = client.get("/api/v1/traces/analytics")
    assert a.status_code == 200
    assert a.json()["n"] >= 1


def test_ml_train_predict(client):
    r = client.post("/api/v1/ml/train", json={"tune": False})
    assert r.status_code == 200, r.text
    pred = client.post(
        "/api/v1/ml/predict",
        json={
            "mode": "batch",
            "records": [
                {
                    "tenure_months": 5,
                    "monthly_charges": 40,
                    "total_charges": 200,
                    "support_tickets": 0,
                    "contract": "one-year",
                }
            ],
        },
    )
    assert pred.status_code == 200, pred.text


def test_dataeng_and_obs(client):
    assert client.post("/api/v1/dataeng/etl").status_code == 200
    assert client.get("/api/v1/dataeng/quality").status_code == 200
    assert client.get("/api/v1/observability/status").status_code == 200
    assert client.get("/api/v1/rag/index").status_code == 200
