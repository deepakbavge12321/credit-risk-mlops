from airflow import DAG
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.email import send_email
from airflow.utils.trigger_rule import TriggerRule

from datetime import datetime, timedelta
import os
import json

# -----------------------------
# Config (IMPORTANT)
# -----------------------------
AIRFLOW_DATA_DIR = "/opt/airflow/data/raw"
PROJECT_DIR = "/opt/project"   # <-- your mounted ML project


# -----------------------------
# 🔹 Alert Functions
# -----------------------------
def no_data_alert(context):
    send_email(
        to="test@airflow.com",
        subject="No New Data Alert",
        html_content="""
        <h3>No new data detected</h3>
        <p>The pipeline did not find any new data today.</p>
        """
    )


def performance_alert():
    metrics_path = os.path.join(PROJECT_DIR, "metrics.json")

    if not os.path.exists(metrics_path):
        raise Exception("metrics.json not found")

    with open(metrics_path) as f:
        m = json.load(f)

    if m["recall"] < 0.30:
        send_email(
            to="test@airflow.com",
            subject="Model Performance Alert",
            html_content=f"""
            <h3>Low Recall Detected</h3>
            <p>Recall: {m['recall']}</p>
            <p>AUC: {m['auc']}</p>
            """
        )
    else:
        print("Model performance OK")


# -----------------------------
# 🔹 Check for NEW data
# -----------------------------
def check_new_data(**context):
    files = [f for f in os.listdir(AIRFLOW_DATA_DIR) if f.endswith(".csv")]

    if not files:
        raise Exception("No CSV files found")

    latest_file = max(
        files,
        key=lambda f: os.path.getmtime(os.path.join(AIRFLOW_DATA_DIR, f))
    )

    print(f"Latest file detected: {latest_file}")
    return latest_file


# -----------------------------
# 🔹 DAG Definition
# -----------------------------
with DAG(
    dag_id="credit_risk_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={
        "owner": "mlops",
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
    }
) as dag:

    # -----------------------------
    # 1. Wait for CSV (Sensor)
    # -----------------------------
    wait_for_data = FileSensor(
        task_id="wait_for_data",
        filepath="data/*.csv",   # relative to AIRFLOW_HOME
        poke_interval=60,
        timeout=600,
        mode="poke",
        on_failure_callback=no_data_alert
    )

    # -----------------------------
    # 2. Check NEW file
    # -----------------------------
    check_data = PythonOperator(
        task_id="check_new_data",
        python_callable=check_new_data,
        on_failure_callback=no_data_alert
    )

    # -----------------------------
    # 3. Run DVC pipeline
    # -----------------------------
    run_pipeline = BashOperator(
        task_id="run_dvc_pipeline",
        bash_command=f"cd {PROJECT_DIR} && dvc repro",
    )

    # -----------------------------
    # 4. Evaluate explicitly
    # -----------------------------
    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command=f"cd {PROJECT_DIR} && python src/evaluate.py",
        trigger_rule=TriggerRule.ALL_DONE
    )

    # -----------------------------
    # 5. Alert
    # -----------------------------
    alert = PythonOperator(
        task_id="performance_alert",
        python_callable=performance_alert,
        trigger_rule=TriggerRule.ALL_DONE
    )

    # -----------------------------
    # DAG Flow
    # -----------------------------
    wait_for_data >> check_data >> run_pipeline >> evaluate >> alert