import mysql.connector

# Connect to the MySQL database
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Rishi@191612",
        database="vishion_app"
    )

def insert_report(username, report_date, title, description, file_blob):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO baby_reports (username, report_date, report_name, description, file)
        VALUES (%s, %s, %s, %s, %s)
    """, (username, report_date, title, description, file_blob))
    conn.commit()
    conn.close()

def add_report(username, report_date, report_name, description):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO baby_reports (username, report_date, report_name, description)
        VALUES (%s, %s, %s, %s)
    """, (username, report_date, report_name, description))
    conn.commit()
    conn.close()

def get_reports(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT report_date, report_name, description
        FROM baby_reports
        WHERE username = %s
        ORDER BY report_date DESC
    """, (username,))
    results = cursor.fetchall()
    conn.close()
    return results
