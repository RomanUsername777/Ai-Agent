"""Главный файл для запуска агента с интерактивным интерфейсом"""
import argparse
import sys
from console_interface import ConsoleInterface


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Агент для автоматизации браузера",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main_agent.py
  python main_agent.py --session mail_session
  python main_agent.py --headless
  python main_agent.py --session hh --headless
        """
    )
    
    parser.add_argument(
        "--session", "-s",
        type=str,
        default=None,
        help="Имя сессии для сохранения (по умолчанию: default)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Запуск браузера в headless режиме (невидимый). По умолчанию браузер видимый."
    )
    
    args = parser.parse_args()
    
    try:
        # Создаем и запускаем интерфейс
        interface = ConsoleInterface(
            session_name=args.session,
            headless=args.headless
        )
        interface.run()
    except KeyboardInterrupt:
        print("\n\n👋 Выход из программы...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

