"""Celery worker entry point: `celery -A data5580_hw.worker worker`.

Tasks use the SQLAlchemy session, so they run inside a Flask app context and the
worker builds the same app the API does (create_app configures the broker and
binds the app to the Celery instance). Importing the tasks module is what
registers the tasks; a worker started from celery_app alone rejects every job
as unregistered.
"""
from data5580_hw.app import create_app
from data5580_hw.celery_app import celery  # noqa: F401  (located by `-A`)
from data5580_hw.tasks import prediction_tasks  # noqa: F401  (registers tasks)

flask_app = create_app()
