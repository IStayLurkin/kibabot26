import sqlite3

def wipe_poisoned_context():
    conn = sqlite3.connect('bot.db')
    cursor = conn.cursor()
    
    # Wipe the recent messages and summaries so he forgets his own lies
    cursor.execute("DELETE FROM chat_messages")
    cursor.execute("DELETE FROM chat_summaries")
    cursor.execute("DELETE FROM chat_state")
    
    conn.commit()
    conn.close()
    print("🧹 Poisoned history wiped. Kiba's short-term memory is clean.")

if __name__ == "__main__":
    wipe_poisoned_context()