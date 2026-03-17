from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error, pooling
import time
import json
import os
import numpy as np
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration from your Railway Credentials
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "metro.proxy.rlwy.net"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", "PDrDouMEHuOQiAbNdJAWEtakbTUuQRKC"),
    "database": os.getenv("DB_NAME", "inventory_system"),
    "port": int(os.getenv("DB_PORT", 3306))
}

# Connection Pool to handle frequent cloud requests
try:
    db_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="velyn_pool",
        pool_size=5,
        **DB_CONFIG
    )
except Error as e:
    print(f"Error creating connection pool: {e}")

ANOMALY_QUEUE = []
LAST_ANOMALY_TIME = 0
COOLDOWN = 60 
BRAIN_FILE = 'velyn_brain.json'

def load_velyn_brain():
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def get_db_connection():
    return db_pool.get_connection()

def calculate_ml_risk(model_name, age_days, failure_count, brain):
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
    
    if score > 0.85:
        status = "Critical"
        rec = "Decommission immediately. This unit is highly likely to fail."
        thoughts = f"High Risk Alert: This {model_name} has a {int(score*100)}% risk of total failure."
    elif score > 0.50:
        status = "Warning"
        rec = "Schedule a maintenance check-up soon."
        thoughts = f"Early Warning: Unusual wear in: {comp_str or 'general components'}. Monitor closely."
    else:
        status = "Healthy"
        rec = "No action needed."
        thoughts = f"System Normal: The {model_name} is performing within parameters for its age ({age_days} days)."

    if age_days <= 14 and d_count >= 1:
        status = "Critical"
        thoughts = f"Defective Unit Detected: This brand-new {model_name} failed within {age_days} days."
        rec = "Contact supplier for a replacement (RMA) immediately."

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
            age_days = (now - datetime.combine(row['date_acquired'], datetime.min.time())).days if row['date_acquired'] else 0
            
            cursor.execute("SELECT component FROM reported_items WHERE asset_id = %s AND status != 'Resolved'", (row['asset_id'],))
            comps = [r['component'] for r in cursor.fetchall()]

            thoughts, status, rec, score = analyze_asset_ml(row['model_name'], len(comps), age_days, comps, brain)
            
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
        
        ANOMALY_QUEUE = sorted(temp_queue, key=lambda x: x['risk_score'], reverse=True)
    except Exception as e: 
        print(f"Error refreshing queue: {e}")
    finally:
        if conn: conn.close()

@app.route('/')
def health_check():
    return "Velyn AI Predictive Engine is LIVE."

@app.route('/scan', methods=['GET'])
def scan_assets():
    global LAST_ANOMALY_TIME, ANOMALY_QUEUE
    current_time = int(time.time())
    scan_type = request.args.get('type', 'greeting')
    response = {"messages": [], "critical": False, "anomaly": None, "strategic_insights": []}

    if scan_type == 'standby':
        brain = load_velyn_brain()
        for model, data in brain.items():
            if data.get('expected_life', 1095) < 547:
                response['strategic_insights'].append({
                    "type": "Procurement Risk",
                    "message": f"Stop buying {model} units. They typically fail after only {data['expected_life']} days."
                })

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
    if not os.path.exists(BRAIN_FILE):
        with open(BRAIN_FILE, 'w') as f:
            json.dump({}, f)
            
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
