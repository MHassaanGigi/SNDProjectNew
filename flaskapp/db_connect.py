# app/db_connect.py
import pyodbc
import sqlite3
import pandas as pd

def get_connection():
    return pyodbc.connect(
        'DRIVER={ODBC Driver 18 for SQL Server};'
        'SERVER=172.19.0.75\\sndpro;'
        'DATABASE=Project;'
        'UID=project;'
        'PWD=P@kistan!@#$;'
        'Encrypt=no;'
    )

def run_query(query, as_dataframe=False):
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df if as_dataframe else df.to_dict(orient='records')
