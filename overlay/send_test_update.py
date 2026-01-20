"""Small helper to POST a test update to the overlay server."""
import requests
import time

URL = "http://localhost:8080/update"

if __name__ == '__main__':
    payload = {
        'stats': {'tiles_visited': 123, 'episode_steps': 42},
        'commentary': 'Test update from send_test_update.py',
        'game_state': {'location': 'Pallet Town', 'x': 10, 'y': 5},
        'action_confidence': {'up':0.1,'down':0.2,'left':0.3,'right':0.15,'a':0.05,'b':0.2}
    }

    try:
        r = requests.post(URL, json=payload, timeout=3)
        print('Response:', r.status_code, r.text)
    except Exception as e:
        print('Failed to send update:', e)
