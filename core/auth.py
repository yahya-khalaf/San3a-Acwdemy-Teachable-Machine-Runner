# core/auth.py
import csv
import requests

class CSVAuth:
    def __init__(self, csv_url: str):
        self.csv_url = csv_url

    def fetch_users(self):
        try:
            response = requests.get(self.csv_url, timeout=5)
            response.raise_for_status()
            content = response.text.splitlines()
            return list(csv.DictReader(content))
        except Exception as e:
            print(f"[auth] Error fetching CSV: {e}")
            return None

    def check_login(self, username: str, password: str) -> bool:
        rows = self.fetch_users()
        if not rows:
            return False

        username = username.strip().lower()

        for row in rows:
            row_user = row.get("username", "").strip().lower()
            row_pass = row.get("password", "").strip()
            status   = row.get("status", "").strip().lower()

            if row_user == username:
                if status != "active":
                    return False
                if row_pass == password:
                    return True
                return False

        return False
