#!/usr/bin/env python3
"""
Test script untuk BISAbot dengan model Llama-3.1-8B-Instruct
"""

from features.bisabot.bisabot import ask_bisabot, get_chat_history, clear_chat_history

def test_bisabot():
    print("=== Test BISAbot dengan Llama-3.1-8B-Instruct ===\n")
    
    # Test pertanyaan pertama
    print("User: Apa itu asuransi kesehatan?")
    response1 = ask_bisabot("Apa itu asuransi kesehatan?")
    print(f"BISAbot: {response1}\n")
    
    # Test pertanyaan lanjutan (dengan context history)
    print("User: Apa saja jenis-jenisnya?")
    response2 = ask_bisabot("Apa saja jenis-jenisnya?")
    print(f"BISAbot: {response2}\n")
    
    # Test pertanyaan ketiga
    print("User: Bagaimana cara memilih yang tepat?")
    response3 = ask_bisabot("Bagaimana cara memilih yang tepat?")
    print(f"BISAbot: {response3}\n")
    
    # Tampilkan history
    print("=== Chat History ===")
    history = get_chat_history()
    for i, msg in enumerate(history):
        role = "User" if msg["role"] == "user" else "BISAbot"
        print(f"{i+1}. {role}: {msg['content'][:100]}...")
    
    print(f"\nTotal messages in history: {len(history)}")
    
    # Test clear history
    print("\n=== Clearing History ===")
    clear_chat_history()
    history_after_clear = get_chat_history()
    print(f"Messages after clear: {len(history_after_clear)}")

if __name__ == "__main__":
    test_bisabot()
