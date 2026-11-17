from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase


class BaseModel(DeclarativeBase):

    pass

db = SQLAlchemy(model_class=BaseModel)
migrate = Migrate()


def init_db(app) -> None:
    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        from data5580_hw.services.database import user_model  # noqa: F401
        from data5580_hw.services.database import prediction  # noqa: F401

        database_url = app.config.get("SQLALCHEMY_DATABASE_URI", "")
        if database_url.startswith("sqlite"):
            # Dev / test: always start with a clean schema
            try:
                db.drop_all()
            except Exception:
                pass
            finally:
                db.create_all()
                db.session.commit()
        elif db.engine.dialect.name == "postgresql":
            # Production: create tables only if they don't exist yet. The gunicorn
            # workers and the Celery worker all start at once against a fresh
            # database, so serialize table creation with an advisory lock.
            with db.engine.begin() as conn:
                conn.execute(text("SELECT pg_advisory_xact_lock(5580)"))
                db.metadata.create_all(bind=conn)
        else:
            db.create_all()
            db.session.commit()
