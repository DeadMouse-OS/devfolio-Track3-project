# Project: Financial Anomaly Detection & Alerting System with AWS + LLM Summarization

This project is an intelligent real-time monitoring agent that fetches live market data from Yahoo Finance, detects multi-dimensional anomalies using statistical and ML methods (Isolation Forest + LOF + Returns), and produces LLM-powered summaries of the detected anomalies.

It then sends alerts to AWS SNS and stores structured results in DynamoDB for audit and analysis.
This system can be deployed as an AWS Lambda container, triggered periodically by EventBridge (cron rule).
