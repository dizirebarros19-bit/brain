from flask import Flask, request, jsonify
import mysql.connector
import time
import json
import os
import numpy as np
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

ANOMALY_QUEUE = []
LAST_ANOMALY_TIME = 0
COOLDOWN = 60 
BRAIN_FILE = 'velyn_brain.json'

def load_velyn_brain():
    if os.path.exists(BRAIN_FILE):
        with open(BRAIN_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_db_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="", database="inventory_system"
    )

def calculate_ml_risk(model_name, age_days, failure_count, brain):
    # Use learned life or default to 3 years (1095 days)
    history = brain.get(model_name, {"expected_life": 1095, "risk_factor": 0.2})
    expected_life = history['expected_life']
    
    life_ratio = age_days / expected_life
    # Sigmoid function for aging risk
    age_risk = 1 / (1 + np.exp(-(life_ratio - 0.8) * 10)) 
    fail_risk = min(failure_count * 0.20, 0.7)
    
    total_score = min(age_risk + fail_risk + history['risk_factor'], 1.0)
    return round(total_score, 2)

def analyze_asset_ml(model_name, d_count, age_days, components_list, brain):
    score = calculate_ml_risk(model_name, age_days, d_count, brain)
    comp_str = ", ".join(components_list[:3]) + ("..." if len(components_list) > 3 else "")
    
    # Human-Readable Logic
    if score > 0.85:
        status = "Critical"
        rec = "Decommission immediately. This unit is highly likely to fail based on historical patterns."
        thoughts = f"High Risk Alert: This {model_name} has a {int(score*100)}% risk of total failure. Immediate replacement recommended."
    elif score > 0.50:
        status = "Warning"
        rec = "Schedule a maintenance check-up soon."
        thoughts = f"Early Warning: We are seeing unusual wear in: {comp_str or 'general components'}. Monitor closely."
    else:
        status = "Healthy"
        rec = "No action needed."
        thoughts = f"System Normal: The {model_name} is performing within expected parameters for its age ({age_days} days)."

    # Special Case: "Lemon" Detection (New items failing immediately)
    if age_days <= 14 and d_count >= 1:
        status = "Critical"
        thoughts = f"Defective Unit Detected: This brand-new {model_name} failed within just {age_days} days of use."
        rec = "Contact supplier for a replacement (RMA) immediately. This is a factory defect ('Lemon')."

    return thoughts, status, rec, score

def refresh_anomaly_queue():
    global ANOMALY_QUEUE
    conn = None
    try:
        brain = load_velyn_brain()
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT a.asset_id, a.date_acquired, c.category_name as model_name
            FROM assets a JOIN asset_categories c ON a.category_id = c.category_id
            WHERE a.deleted = 0
        """)
        rows = cursor.fetchall()
        
        temp_queue = []
        now = datetime.now()
        for row in rows:
            # Calculate age
            age_days = (now - datetime.combine(row['date_acquired'], datetime.min.time())).days if row['date_acquired'] else 0
            
            # Check for active (unresolved) reports
            cursor.execute("SELECT component FROM reported_items WHERE asset_id = %s AND status != 'Resolved'", (row['asset_id'],))
            comps = [r['component'] for r in cursor.fetchall()]

            thoughts, status, rec, score = analyze_asset_ml(row['model_name'], len(comps), age_days, comps, brain)
            
            # Only add to queue if there is a genuine concern
            if score > 0.3 or status != "Healthy":
                temp_queue.append({
                    'asset_id': row['asset_id'], 
                    'model': row['model_name'],
                    'thoughts': thoughts, 
                    'recommendation': rec,
                    'risk_score': score, 
                    'severity': status, 
                    'age_days': age_days
                })
        
        # Sort by highest risk first
        ANOMALY_QUEUE = sorted(temp_queue, key=lambda x: x['risk_score'], reverse=True)
    except Exception as e: 
        print(f"Error refreshing queue: {e}")
    finally:
        if conn: conn.close()

@app.route('/scan', methods=['GET'])
def scan_assets():
    global LAST_ANOMALY_TIME, ANOMALY_QUEUE
    current_time = int(time.time())
    scan_type = request.args.get('type', 'greeting')
    response = {"messages": [], "critical": False, "anomaly": None, "strategic_insights": []}

    if scan_type == 'standby':
        # 1. Check Strategic Insights (Procurement Advice)
        brain = load_velyn_brain()
        for model, data in brain.items():
            if data.get('expected_life', 1095) < 547:
                response['strategic_insights'].append({
                    "type": "Procurement Risk",
                    "message": f"Stop buying {model} units. They typically fail after only {data['expected_life']} days."
                })

        # 2. Check for Active Anomalies
        if not ANOMALY_QUEUE: 
            refresh_anomaly_queue()
            
        if ANOMALY_QUEUE and (current_time - LAST_ANOMALY_TIME >= COOLDOWN):
            anomaly = ANOMALY_QUEUE.pop(0)
            LAST_ANOMALY_TIME = current_time
            response['critical'] = True
            response['anomaly'] = anomaly
    else:
        response['messages'].append("Velyn AI Predictive Engine is online and monitoring assets.")
    
    return jsonify(response)

@app.route('/all_anomalies', methods=['GET'])
def all_anomalies():
    refresh_anomaly_queue()
    return jsonify({"anomalies": ANOMALY_QUEUE, "count": len(ANOMALY_QUEUE)})

if __name__ == '__main__':
    # Ensure brain file exists to avoid errors on first run
    if not os.path.exists(BRAIN_FILE):
        with open(BRAIN_FILE, 'w') as f:
            json.dump({}, f)
            
    app.run(port=5000, debug=True)