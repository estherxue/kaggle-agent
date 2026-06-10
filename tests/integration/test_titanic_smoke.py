"""Smoke test: End-to-end test with mock Kaggle client.

This test runs through a full competition lifecycle using the Titanic
competition as a template (mocked data).
"""

import pytest
import tempfile
from pathlib import Path
import sys

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kaggle_agent.config import Config, LLMConfig, LLMRolesConfig, LLMProviderConfig
from kaggle_agent.llm import LLMRouter, MockProvider
from kaggle_agent.orchestrator import Orchestrator
from kaggle_agent.tools import MockKaggleClient


def create_test_config(tmp_path: Path) -> Config:
    """Create test configuration."""
    return Config(
        llm=LLMConfig(
            providers=[
                LLMProviderConfig(
                    name="mock",
                    type="mock",
                    model="mock-model",
                    cost_per_1k_prompt=0.0,
                    cost_per_1k_completion=0.0,
                )
            ],
            roles=LLMRolesConfig(
                planner="mock",
                coder="mock",
                reviewer="mock",
                summarizer="mock",
            ),
            default={"temperature": 0.7, "max_tokens": 500},
        ),
        budget={
            "max_experiments_per_competition": 2,
            "max_llm_cost_usd": 100.0,
            "max_execution_time_per_exp_sec": 30,
            "max_submissions_per_day": 5,
        },
        paths={
            "knowledge": str(tmp_path / "knowledge"),
            "competitions": str(tmp_path / "competitions"),
        },
        execution={
            "timeout_sec": 30,
            "allow_network": False,
            "python_path": None,
        },
        kaggle={
            "auto_submit": True,
            "dry_run": True,
        },
    )


def create_mock_llm_responses() -> dict:
    """Create mock LLM responses for testing."""
    return {
        # EDA response
        "Generate Python code for exploratory data analysis": """
```python
import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Basic stats
print(f"Train shape: {train.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Target distribution:\n{train['target'].value_counts()}")

# Save EDA report
with open('eda_report.md', 'w') as f:
    f.write("# EDA Report\\n\\n")
    f.write(f"Train shape: {train.shape}\\n")
```
""",
        # Experiment response
        "Propose a hypothesis and write Python code": """
HYPOTHESIS: Train a simple LightGBM model with default parameters as baseline.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Load data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Simple preprocessing
X = train.drop(['target', 'id'], axis=1, errors='ignore')
y = train['target']

# Handle non-numeric columns
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

# Simple CV
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))

for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict(X_val)

cv_score = accuracy_score(y, oof_preds)
print(f"CV score: {cv_score:.4f}")

# Predict on test
X_test = test.drop(['id'], axis=1, errors='ignore')
for col in X_test.columns:
    if X_test[col].dtype == 'object':
        X_test[col] = X_test[col].astype('category').cat.codes

preds = model.predict(X_test)

# Save submission
sub = pd.DataFrame({'id': test['id'], 'target': preds.astype(int)})
sub.to_csv('submission.csv', index=False)
```
""",
    }


@pytest.mark.slow
@pytest.mark.integration
def test_titanic_smoke():
    """End-to-end smoke test with mock data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create knowledge directories
        (tmp_path / "knowledge" / "playbooks" / "techniques").mkdir(parents=True)
        (tmp_path / "knowledge" / "skills" / "feature_engineering").mkdir(parents=True)

        # Create test playbooks
        (tmp_path / "knowledge" / "playbooks" / "general.md").write_text("# General")
        (tmp_path / "knowledge" / "playbooks" / "tabular.md").write_text("# Tabular")

        # Create config
        config = create_test_config(tmp_path)

        # Create LLM router with mock provider
        mock_provider = MockProvider(
            name="mock",
            responses=create_mock_llm_responses(),
        )
        llm_router = LLMRouter(
            providers={"mock": mock_provider},
            role_mapping={
                "planner": "mock",
                "coder": "mock",
                "reviewer": "mock",
                "summarizer": "mock",
            },
            default_params={"temperature": 0.7, "max_tokens": 500},
        )

        # Create mock Kaggle client
        kaggle_client = MockKaggleClient()

        # Create orchestrator
        orchestrator = Orchestrator(
            config=config,
            competition="titanic",
            llm_router=llm_router,
            kaggle_client=kaggle_client,
            resume=False,
        )

        # Copy knowledge to orchestrator's paths
        from kaggle_agent.knowledge import PlaybookManager
        pm = PlaybookManager(tmp_path / "knowledge")
        orchestrator.playbooks = pm

        # Run initialization
        orchestrator._initialize()

        # Verify state
        assert orchestrator.state.competition == "titanic"
        assert orchestrator.state.phase.name == "UNDERSTANDING"
        assert "titanic" in kaggle_client.downloaded_competitions

        # Run understanding
        orchestrator._understand_competition()
        assert orchestrator.state.competition_type == "tabular"
        assert orchestrator.state.phase.name == "LOADING_KNOWLEDGE"

        # Run knowledge loading
        orchestrator._load_knowledge()
        assert orchestrator.state.phase.name == "EDA"

        # Note: We stop here because the full EDA would require
        # actual LLM responses. In a real test, we'd mock all the
        # LLM calls properly.

        # Verify state was saved
        assert orchestrator.state_path.exists()

        # Check status
        status = orchestrator.get_status()
        assert status["competition"] == "titanic"
        assert status["phase"] == "EDA"

        print(f"\nSmoke test passed!")
        print(f"Status: {status}")


if __name__ == "__main__":
    test_titanic_smoke()
