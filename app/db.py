import os
from contextlib import contextmanager
from datetime import datetime
try:
    import mysql.connector
except ImportError:
    mysql = None
else:
    mysql = mysql.connector
from dotenv import load_dotenv
from pathlib import Path
from .security import hash_password

BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / '.env')


def config():
    return {
        'host': os.getenv('MYSQL_HOST', '127.0.0.1'),
        'port': int(os.getenv('MYSQL_PORT', '3306')),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', 'Tony@2404'),
        'database': os.getenv('MYSQL_DATABASE', 'churn_db'),
    }

@contextmanager
def connection():
    if mysql is None: raise RuntimeError('mysql-connector-python is not installed. Run: pip install -r requirements.txt')
    conn = mysql.connect(**config())
    try:
        yield conn
    finally:
        if conn.is_connected():
            conn.close()


def ensure_database():
    cfg = config(); database = cfg.pop('database')
    if mysql is None: raise RuntimeError('mysql-connector-python is not installed. Run: pip install -r requirements.txt')
    conn = mysql.connect(**cfg)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    cur.close(); conn.close()
    with connection() as conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(80) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(30) NOT NULL DEFAULT 'admin',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS customers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(100) NOT NULL UNIQUE,
            gender VARCHAR(30), senior_citizen INT, partner VARCHAR(20), dependents VARCHAR(20),
            tenure INT, phone_service VARCHAR(30), multiple_lines VARCHAR(40), service_type VARCHAR(40),
            online_security VARCHAR(30), online_backup VARCHAR(30), device_protection VARCHAR(30), tech_support VARCHAR(30),
            streaming_tv VARCHAR(30), streaming_movies VARCHAR(30), contract_type VARCHAR(40), paperless_billing VARCHAR(20),
            payment_method VARCHAR(50), monthly_charges DECIMAL(12,2), total_charges DECIMAL(12,2),
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(100) NOT NULL,
            churn_probability DECIMAL(8,6) NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            recommendation TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_prediction_customer (customer_id),
            INDEX idx_prediction_created (created_at)
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS retention_actions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(100) NOT NULL,
            action VARCHAR(255) NOT NULL,
            notes TEXT,
            status VARCHAR(30) NOT NULL DEFAULT 'Planned',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )''')
        username = os.getenv('ADMIN_USERNAME', 'admin')
        password = os.getenv('ADMIN_PASSWORD', 'ChangeMe123!')
        cur.execute('SELECT id FROM users WHERE username=%s', (username,))
        if cur.fetchone() is None:
            cur.execute('INSERT INTO users(username,password_hash,role) VALUES(%s,%s,%s)', (username, hash_password(password), 'admin'))
        conn.commit(); cur.close()


def db_status():
    try:
        with connection() as conn:
            return {'ok': conn.is_connected(), 'message': 'MySQL connected'}
    except Exception as exc:
        return {'ok': False, 'message': str(exc)}


def authenticate(username, password, verify):
    with connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute('SELECT * FROM users WHERE username=%s LIMIT 1', (username,))
        row = cur.fetchone(); cur.close()
    return row if row and verify(password, row['password_hash']) else None


def save_customer(data):
    columns = ['customer_id','gender','senior_citizen','partner','dependents','tenure','phone_service','multiple_lines','service_type','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','contract_type','paperless_billing','payment_method','monthly_charges','total_charges']
    vals = [data.get(c) for c in columns]
    placeholders = ','.join(['%s'] * len(columns))
    updates = ','.join(f'{c}=VALUES({c})' for c in columns[1:])
    with connection() as conn:
        cur = conn.cursor()
        cur.execute(f'INSERT INTO customers({",".join(columns)}) VALUES({placeholders}) ON DUPLICATE KEY UPDATE {updates}', vals)
        conn.commit(); cur.close()



def list_customers(q='', limit=100):
    sql = 'SELECT * FROM customers WHERE 1=1'
    params = []
    if q:
        sql += ' AND customer_id LIKE %s'
        params.append(f'%{q}%')
    sql += ' ORDER BY updated_at DESC LIMIT %s'
    params.append(limit)
    with connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows


def get_customer(customer_id):
    with connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute('SELECT * FROM customers WHERE customer_id=%s LIMIT 1', (customer_id,))
        row = cur.fetchone()
        cur.close()
        return row


def latest_prediction(customer_id):
    with connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute('SELECT * FROM predictions WHERE customer_id=%s ORDER BY created_at DESC LIMIT 1', (customer_id,))
        row = cur.fetchone()
        cur.close()
        return row

def prediction_history(customer_id, limit=30):
    with connection() as conn:
        cur=conn.cursor(dictionary=True)
        cur.execute('SELECT id, customer_id, churn_probability, risk_level, recommendation, created_at FROM predictions WHERE customer_id=%s ORDER BY created_at ASC LIMIT %s',(customer_id, limit))
        rows=cur.fetchall(); cur.close(); return rows

def save_prediction(customer_id, probability, risk, recommendation):
    with connection() as conn:
        cur=conn.cursor()
        cur.execute('INSERT INTO predictions(customer_id,churn_probability,risk_level,recommendation) VALUES(%s,%s,%s,%s)', (customer_id, probability, risk, recommendation))
        conn.commit(); new_id=cur.lastrowid; cur.close(); return new_id


def recent_predictions(limit=50):
    with connection() as conn:
        cur=conn.cursor(dictionary=True)
        cur.execute('SELECT * FROM predictions ORDER BY created_at DESC LIMIT %s', (limit,))
        rows=cur.fetchall(); cur.close(); return rows


def search_predictions(q='', risk='', limit=100):
    sql='SELECT * FROM predictions WHERE 1=1'; params=[]
    if q:
        sql+=' AND customer_id LIKE %s'; params.append(f'%{q}%')
    if risk:
        sql+=' AND risk_level=%s'; params.append(risk)
    sql+=' ORDER BY created_at DESC LIMIT %s'; params.append(limit)
    with connection() as conn:
        cur=conn.cursor(dictionary=True); cur.execute(sql, params); rows=cur.fetchall(); cur.close(); return rows


def prediction_stats():
    with connection() as conn:
        cur=conn.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) total, SUM(risk_level='HIGH') high_risk, SUM(risk_level='MEDIUM') medium_risk, SUM(risk_level='LOW') low_risk, AVG(churn_probability) avg_probability FROM predictions")
        row=cur.fetchone(); cur.close(); return row


def save_retention_action(customer_id, action, notes, status='Planned'):
    with connection() as conn:
        cur=conn.cursor(); cur.execute('INSERT INTO retention_actions(customer_id,action,notes,status) VALUES(%s,%s,%s,%s)', (customer_id,action,notes,status)); conn.commit(); cur.close()


def recent_actions(limit=50):
    with connection() as conn:
        cur=conn.cursor(dictionary=True); cur.execute('SELECT * FROM retention_actions ORDER BY created_at DESC LIMIT %s',(limit,)); rows=cur.fetchall(); cur.close(); return rows
