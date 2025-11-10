# app.py
import os
import json
import time
import uuid
import logging
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# --------- Configuration from env ----------
TABLE_NAME = os.environ.get("TABLE_NAME", "FinancialAnomalies")
TOPIC_ARN = os.environ.get("TOPIC_ARN")  # SNS FIFO or standard
TICKER = os.environ.get("TICKER", "AAPL")
FETCH_PERIOD = os.environ.get("FETCH_PERIOD", "2d")  # e.g., "1d","2d"
FETCH_INTERVAL = os.environ.get("FETCH_INTERVAL", "1m")  # "1m", "5m"
ROLLING_WINDOW = int(os.environ.get("ROLLING_WINDOW", "20"))  # window for rolling stats
CONTAMINATION = float(os.environ.get("CONTAMINATION", "0.02"))  # for iso forest
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-embed-text")  # replace with textual model id

# --------- AWS clients ----------
dynamodb = boto3.resource("dynamodb")
sns = boto3.client("sns")
# Bedrock runtime client (requires aws-sdk version that supports bedrock-runtime)
bedrock = boto3.client("bedrock-runtime")  # may require correct region/permissions

# --------- Logging ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# --------- Utilities ----------
def fetch_data(ticker=TICKER, period=FETCH_PERIOD, interval=FETCH_INTERVAL):
    """
    Fetch intraday data via yfinance. Returns pandas DataFrame with Datetime index.
    """
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # standardize column names
    df.rename(columns={"Datetime": "Datetime", "Open": "Open", "High": "High",
                       "Low": "Low", "Close": "Close", "Volume": "Volume"}, inplace=True)
    return df


def add_derived_features(df):
    """
    Add returns, HL gap, OC gap and rolling volatility.
    """
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["hl_gap"] = df["High"] - df["Low"]
    df["oc_gap"] = df["Open"] - df["Close"]
    df["volatility"] = df["return"].rolling(ROLLING_WINDOW).std()
    df.dropna(inplace=True)
    return df


def rolling_standardize_single_point(df_window, features):
    """
    Given a DataFrame window (shape >= 1), fit a StandardScaler on the window (excluding
    the last row optionally) and return scaled last-row feature vector. We'll scale using
    the previous window (past data) then compute features for the newest point.
    """
    scaler = StandardScaler()
    X = df_window[features].values
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled  # returns all rows scaled (we will take last)


def ensemble_anomaly_score(X_window, contamination=CONTAMINATION):
    """
    Fit lightweight ensemble detectors on the window and return anomaly flags for the last row.
    We fit on the whole window to learn normal behavior and mark the points that are outliers.
    """
    # Use IsolationForest + LocalOutlierFactor (LOF)
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)

    # Fit iso on X and get scores
    iso.fit(X_window)
    iso_scores = -iso.score_samples(X_window)  # higher => more anomalous

    # LOF is unsupervised; fit_predict returns -1 for outliers
    lof_labels = lof.fit_predict(X_window)
    # Convert LOF to numeric anomaly score: outlier (-1) -> 1, normal (1) -> 0
    lof_scores = (lof_labels == -1).astype(float)

    # Combine scores: normalize iso_scores to 0-1
    iso_norm = (iso_scores - np.min(iso_scores)) / (np.max(iso_scores) - np.min(iso_scores) + 1e-9)
    combined = 0.6 * iso_norm + 0.4 * lof_scores  # weighted ensemble

    return combined  # array same length as X_window; larger => more anomalous


def detect_latest_anomaly(df, features):
    """
    Use rolling standardization + ensemble detection on the most recent window and decide
    whether latest row is anomalous. Returns anomaly dict if anomalous, else None.
    """
    if len(df) < ROLLING_WINDOW + 1:
        return None

    # select last window rows (we include last)
    window_df = df.iloc[-(ROLLING_WINDOW + 1):].copy().reset_index(drop=True)
    # standardize on the window except: fit on all rows (this simulates using only past info in real streaming)
    X_scaled = rolling_standardize_single_point(window_df, features)
    scores = ensemble_anomaly_score(X_scaled, contamination=CONTAMINATION)

    latest_score = float(scores[-1])
    # define threshold: choose percentiles dynamically
    thresh = np.percentile(scores, 100 * (1 - CONTAMINATION))
    is_anomaly = latest_score >= thresh

    if not is_anomaly:
        return None

    # prepare anomaly details for latest point
    latest_row = window_df.iloc[-1].to_dict()
    return {
        "score": latest_score,
        "threshold": float(thresh),
        "latest": latest_row,
        "window_stats": {
            "mean_scores": float(np.mean(scores)),
            "max_score": float(np.max(scores))
        }
    }


def build_event_payload(ticker, anomaly, df):
    """
    Build structured JSON context for Bedrock summarizer and for logging.
    """
    latest = anomaly["latest"]
    ts = latest.get("Datetime") if "Datetime" in latest else latest.get("datetime", None)
    if isinstance(ts, (pd.Timestamp, )):
        ts = ts.isoformat()
    payload = {
        "timestamp": ts,
        "ticker": ticker,
        "anomaly_score": anomaly["score"],
        "threshold": anomaly["threshold"],
        "price": {
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"])
        },
        "volume": float(latest["Volume"]),
        "return": float(latest.get("return", 0.0)),
        "hl_gap": float(latest.get("hl_gap", 0.0)),
        "oc_gap": float(latest.get("oc_gap", 0.0)),
        "context_summary": f"rolling_window={ROLLING_WINDOW}"
    }
    return payload


def call_bedrock_summarizer(event_payload, model_id=BEDROCK_MODEL_ID, temperature=0.0):
    """
    Call Bedrock runtime to summarize event_payload. This code uses the bedrock-runtime invoke_model
    interface. Adjust the request/response parsing for the model you pick.
    NOTE: You must have permissions to call bedrock-runtime and valid model id in your account.
    """
    # Build a human-friendly prompt
    prompt = f"""
You are a concise professional financial analyst. Given this event JSON, summarize in 2-3 sentences:
Event: {json.dumps(event_payload)}
Produce: a short title (7-10 words) and then a concise one-sentence explanation of likely cause & recommendation.
Return JSON with keys: title, summary, severity(1-10).
"""
    try:
        # Bedrock runtime expects a specific content-type and model Id. This code may need small edits
        # depending on the Bedrock model you select. See AWS Bedrock docs for exact fields.
        resp = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({"input": prompt, "temperature": temperature}),
            contentType="application/json",
            accept="application/json"
        )
        body_bytes = resp["body"].read()
        text = body_bytes.decode("utf-8")
        # many Bedrock models return text; assume they return a JSON-like text. Try parse, else fallback to raw.
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            # fallback: return the raw model text as summary
            return {"title": "Anomaly detected", "summary": text, "severity": 5}
    except Exception as e:
        logger.exception("Bedrock summarizer failed")
        return {"title": "LLM error", "summary": f"Bedrock error: {e}", "severity": 5}


def put_to_dynamo(table_name, event_payload, summary):
    """
    Write anomaly record to DynamoDB table.
    Table schema: partition key -> ticker (string); sort key -> timestamp (string)
    """
    table = dynamodb.Table(table_name)
    item = {
        "ticker": event_payload["ticker"],
        "timestamp": event_payload["timestamp"],
        "price": event_payload["price"],
        "volume": Decimal_or_float(event_payload["volume"]),
        "return": Decimal_or_float(event_payload["return"]),
        "anomaly_score": Decimal_or_float(event_payload["anomaly_score"]),
        "summary_title": summary.get("title"),
        "summary_text": summary.get("summary"),
        "severity": Decimal_or_float(summary.get("severity", 5))
    }
    table.put_item(Item=item)


def Decimal_or_float(v):
    # DynamoDB requires decimal for exactness; boto3 can accept floats but if you prefer Decimal convert.
    # To keep code simple, we'll return float.
    return float(v)


def publish_sns(topic_arn, message, message_group_id=None):
    kwargs = {"TopicArn": topic_arn, "Message": message, "Subject": "Anomaly Alert"}
    if topic_arn.endswith(".fifo"):
        # FIFO requires MessageGroupId and (optionally) DeduplicationId
        kwargs["MessageGroupId"] = message_group_id or "default"
        kwargs["MessageDeduplicationId"] = str(uuid.uuid4())
    resp = sns.publish(**kwargs)
    return resp


# ---------- Lambda handler ----------
def lambda_handler(event, context):
    """
    Main entrypoint — can be called by EventBridge scheduled rule with payload {"ticker": "AAPL"}.
    """
    ticker = event.get("ticker", TICKER) if isinstance(event, dict) else TICKER
    logger.info(f"Running anomaly check for {ticker} at {datetime.now(timezone.utc).isoformat()}")

    # Step 1: fetch and prepare
    df = fetch_data(ticker=ticker)
    if df.empty:
        logger.warning("No data fetched")
        return {"status": "no_data"}

    df = add_derived_features(df)
    if df.empty:
        logger.warning("No usable data after feature engineering")
        return {"status": "no_data_post_fe"}

    # Step 2: features to use
    features = ["Open", "High", "Low", "Close", "Volume", "return", "hl_gap", "oc_gap", "volatility"]
    # ensure features exist
    available_feats = [f for f in features if f in df.columns]
    anomaly = detect_latest_anomaly(df[available_feats], available_feats)
    if not anomaly:
        logger.info("No anomaly found.")
        return {"status": "no_anomaly"}

    # Step 3: build event payload and summarize via Bedrock
    event_payload = build_event_payload(ticker, anomaly, df)
    summary = call_bedrock_summarizer(event_payload)

    # Step 4: store and publish
    try:
        put_to_dynamo(TABLE_NAME, event_payload, summary)
    except Exception as e:
        logger.exception("Failed to write to DynamoDB")

    try:
        message = f"{summary.get('title','Anomaly')} | {summary.get('summary')}"
        publish_sns(TOPIC_ARN, message, message_group_id=ticker)
    except Exception:
        logger.exception("Failed to publish SNS message")

    return {"status": "anomaly_reported", "summary": summary}
