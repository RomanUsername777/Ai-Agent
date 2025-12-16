"""
Универсальный скрипт для запуска агента с любой задачей
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from agent import Agent, Browser, ChatOpenAI, ChatAnthropic

load_dotenv()

async def run_agent(task: str, session_dir: str = None, headless: bool = False):
    """Запустить агента с задачей"""
    
    # Выбор LLM провайдера
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if openai_key:
        print("Используем OpenAI")
        llm = ChatOpenAI(model='gpt-4o-mini')
    elif anthropic_key:
        print("Используем Anthropic")
        llm = ChatAnthropic(model='claude-sonnet-4-5-20250929')
    else:
        raise ValueError("Необходим OPENAI_API_KEY или ANTHROPIC_API_KEY в .env файле")
    
    # Определяем файл для сохранения сессии (cookies, localStorage)
    # Используем единый файл сессии для всех задач - агент универсальный
    if session_dir:
        storage_state_path = Path(session_dir)
        if storage_state_path.is_dir():
            # Если указана папка, создаем файл storage_state.json внутри
            storage_state_path = storage_state_path / 'storage_state.json'
    else:
        # Универсальный файл сессии для всех задач
        storage_state_path = Path('./browser_storage_state.json')
    
    print(f"\nСессия будет сохранена в: {storage_state_path}")
    print("(только cookies и localStorage, ~10-50KB)")
    
    # Создаем браузер БЕЗ user_data_dir (чтобы не сохранять весь профиль)
    # Используем только storage_state для сохранения сессии
    from agent.browser import BrowserProfile
    
    browser_profile = BrowserProfile(
        storage_state=str(storage_state_path),
        user_data_dir=None,  # Не сохраняем полный профиль
    )
    
    browser = Browser(
        headless=headless,
        browser_profile=browser_profile,
    )
    
    print("=" * 80)
    print("ЗАПУСК АГЕНТА")
    print("=" * 80)
    print(f"\nЗадача: {task}\n")
    
    # Создаем агента
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        use_vision=True,
        max_actions_per_step=3,
    )
    
    try:
        print("Запускаем агента...\n")
        result = await agent.run(max_steps=50)
        
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТ")
        print("=" * 80)
        
        if result and result.history:
            final_result = result.final_result()
            if final_result:
                print(f"\n{final_result}")
            
            print(f"\nВсего шагов: {len(result.history.history)}")
            print(f"Успешно: {result.is_successful()}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        print(f"Сессия сохранена в: {storage_state_path}")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nЗакрываем браузер...")
        try:
            await browser.close()
        except:
            pass

def main():
    """Главная функция"""
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        # Задача передана как аргумент
        task = ' '.join(sys.argv[1:])
        session_dir = None
        if '--session' in sys.argv:
            idx = sys.argv.index('--session')
            session_dir = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        headless = '--headless' in sys.argv
    else:
        # Интерактивный режим
        print("=" * 80)
        print("УНИВЕРСАЛЬНЫЙ АГЕНТ")
        print("=" * 80)
        print("\nВведите задачу для агента:")
        task = input("> ").strip()
        
        if not task:
            print("Задача не может быть пустой!")
            sys.exit(1)
        
        print("\nОпции:")
        print("1. Использовать сохраненную сессию (введите путь или Enter для автоопределения)")
        session_input = input("   Сессия: ").strip()
        session_dir = session_input if session_input else None
        
        headless_input = input("2. Headless режим? (y/n, по умолчанию n): ").strip().lower()
        headless = headless_input == 'y'
    
    asyncio.run(run_agent(task, session_dir, headless))

if __name__ == '__main__':
    main()

