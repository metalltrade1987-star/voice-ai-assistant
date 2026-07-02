"""Точка входа в приложение"""
from core.assistant import Assistant


if __name__ == "__main__":
    try:
        Assistant().run()
    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
        input("Нажмите Enter...")