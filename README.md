# Project: Financial Anomaly Detection & Alerting System with AWS + LLM Summarization

This project is an intelligent real-time monitoring agent that fetches live market data from Yahoo Finance, detects multi-dimensional anomalies using statistical and ML methods (Isolation Forest + LOF + Returns), and produces LLM-powered summaries of the detected anomalies.

It then sends alerts to AWS SNS and stores structured results in DynamoDB for audit and analysis.
This system can be deployed as an AWS Lambda container, triggered periodically by EventBridge (cron rule).

This project is a real-time financial anomaly detection and alerting system built using:

Yahoo Finance API for live financial data
Isolation Forest & Local Outlier Factor for ensemble anomaly detection
Rolling StandardScaler for dynamic normalization
LLM (BERT / AWS Bedrock) for intelligent natural-language summaries
AWS DynamoDB for anomaly record storage
AWS SNS (FIFO Topic) for automated alert notifications

The system continuously monitors stock data (e.g., AAPL) and automatically:
Detects unusual patterns or outliers (e.g., price jumps, volume drops)
Summarizes anomalies using an LLM
Stores anomaly data in DynamoDB
Sends human-readable alerts through AWS SNS
Generates and stores structured reports in S3 for visualization or audit

