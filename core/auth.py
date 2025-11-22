# core/auth.py
import csv
import requests

class CSVAuth:
    def __init__(self, csv_url: str):
        self.csv_url = csv_url

    def fetch_users(self):
        """
        Fetches users from CSV.
        Returns: (list_of_rows, error_message)
        """
        try:
            response = requests.get(self.csv_url, timeout=5)
            response.raise_for_status()
            content = response.text.splitlines()
            return list(csv.DictReader(content)), None
        except requests.exceptions.ConnectionError:
            return None, "Network error: Check your internet connection."
        except requests.exceptions.Timeout:
            return None, "Connection timed out."
        except Exception as e:
            print(f"[auth] Error fetching CSV: {e}")
            return None, f"Error fetching user database: {e}"

    def check_login(self, username: str, password: str) -> tuple[bool, str]:
        """
        Checks login credentials.
        Returns: (Success: bool, Message: str)
        """
        rows, error_msg = self.fetch_users()
        
        if rows is None:
            return False, error_msg

        username = username.strip().lower()

        user_found = False
        for row in rows:
            row_user = row.get("username", "").strip().lower()
            row_pass = row.get("password", "").strip()
            status   = row.get("status", "").strip().lower()

            if row_user == username:
                user_found = True
                if status != "active":
                    return False, "Account is inactive."
                if row_pass == password:
                    return True, "Login successful."
                return False, "Invalid password."

        if not user_found:
            return False, "Username not found."
        
        return False, "Unknown login error."