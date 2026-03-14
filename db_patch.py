import sqlite3

def patch_db():
    conn = sqlite3.connect('bot.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE runtime_settings SET setting_value = 'kiba' WHERE setting_key = 'active_llm_model'")
    conn.commit()
    conn.close()
    print("Database patched! Qwen3 is gone. Kiba is active.")

if __name__ == "__main__":
    patch_db()