"""
retrain_dag.py
--------------
Airflow DAG for weekly retraining of the recommendation models.

Note: This file is meant to be placed in an Airflow environment.
If you don't have Airflow installed locally, the import errors in the IDE
are expected and can be ignored for the internship demo.
"""

from datetime import datetime, timedelta

# These imports only work when Airflow is installed
try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:
    # Allows the file to exist without breaking local development
    DAG = None
    BashOperator = None


# Default arguments for the DAG
default_args = {
    "owner": "recommendation-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Only create the DAG if Airflow is available
if DAG is not None:
    with DAG(
        dag_id="recommendation_retrain",
        default_args=default_args,
        description="Weekly retrain of the Two-Tower + Ranking models",
        schedule_interval="@weekly",
        start_date=datetime(2026, 11, 1),
        catchup=False,
        tags=["recommendation", "ml", "retrain"],
    ) as dag:

        retrain_task = BashOperator(
            task_id="run_full_retrain",
            bash_command=(
                "cd /path/to/your/recommendation-engine && "
                "python -m src.pipeline.retrain"
            ),
            # ⚠️ Change the path above to your actual project path
        )