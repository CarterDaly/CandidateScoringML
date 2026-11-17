"""Exploration decays with recorded outcomes, not with every prediction."""
from datetime import datetime, timezone

import pytest

from data5580_hw.app import create_app
from data5580_hw.controllers.candidate import _count_feedback
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ModelSQL, PredictionSQL


@pytest.fixture
def app():
    a = create_app()
    a.config["TESTING"] = True
    return a


def _prediction(pid, actual):
    return PredictionSQL(id=pid, features={"f": 1.0}, tags={}, label=0.5, actual=actual, model_id="m-1")


def test_unlabeled_predictions_are_not_counted_as_feedback(app):
    with app.app_context():
        now = datetime.now(timezone.utc)
        db.session.add(ModelSQL(id="m-1", model_type="tree", model_name="m", model_version="1", created=now, updated=now))
        db.session.add_all([_prediction("p-1", None), _prediction("p-2", None), _prediction("p-3", 1)])
        db.session.commit()

        assert _count_feedback("m-1") == 1
