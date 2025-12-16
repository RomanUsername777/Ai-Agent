"""
Тест задачи с Яндекс Почтой: чтение и удаление спама
Требует сохраненную сессию (user_data_dir с авторизацией)
"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from agent import Agent, Browser, ChatOpenAI, ChatAnthropic
from agent.browser.profile import ViewportSize

load_dotenv()

async def test_yandex_mail():
    """Тест чтения и удаления спама в Яндекс Почте"""
    
    # Выбор LLM провайдера
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if openai_key:
        print("Используем OpenAI")
        base_url = os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_URL')
        # Если указан OPENAI_API_URL с полным путём /chat/completions, убираем его
        if base_url and '/chat/completions' in base_url:
            base_url = base_url.replace('/chat/completions', '')
        llm = ChatOpenAI(model='gpt-4o-mini', api_key=openai_key, base_url=base_url)
    elif anthropic_key:
        print("Используем Anthropic")
        llm = ChatAnthropic(model='claude-sonnet-4-5-20250929')
    else:
        # Пробуем использовать ChatBrowserUse (облачный сервис агента, требует AGENT_API_KEY)
        try:
            from agent.llm.agent.chat import ChatBrowserUse
            agent_key = os.getenv('AGENT_API_KEY')
            if agent_key:
                print("Используем ChatBrowserUse")
                llm = ChatBrowserUse()
            else:
                print("⚠️  Не найден API ключ. Пробуем запустить без LLM ключа...")
                print("Если не сработает, установите OPENAI_API_KEY, ANTHROPIC_API_KEY или AGENT_API_KEY")
                # Попробуем все равно запустить - может быть есть дефолтный ключ
                try:
                    llm = ChatBrowserUse()
                except:
                    raise ValueError("Необходим OPENAI_API_KEY, ANTHROPIC_API_KEY или AGENT_API_KEY")
        except ImportError:
            raise ValueError("Необходим OPENAI_API_KEY или ANTHROPIC_API_KEY (ChatBrowserUse недоступен)")
    
    # Используем сохраненную сессию (storage state в формате Chromium)
    storage_state_path = Path('./yandex_mail_storage.json')
    user_data_dir = Path('./yandex_mail_session')
    
    # Создаем браузер с сохранением сессии
    # Размер окна: чуть больше половины экрана (примерно 60% от стандартного)
    browser = Browser(
        headless=False,  # Видимый браузер для наблюдения
        user_data_dir=str(user_data_dir),  # Сохранение сессии
        window_size=ViewportSize(width=1200, height=700),  # Размер окна браузера
    )
    
    # Если есть storage state, загрузим его после запуска браузера
    # Профиль браузера может загружать сохранённое состояние сессии (cookies/localStorage)
    
    # Задача: прочитать последние 10 писем и удалить спам
    task = "Прочитай последние 10 писем в Яндекс Почте и удали спам-письма"
    
    print("=" * 80)
    print("ТЕСТ: Чтение и удаление спама в Яндекс Почте")
    print("=" * 80)
    print(f"\nЗадача: {task}\n")
    print(f"Сессия: {user_data_dir}\n")
    
    # Создаем агента
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        use_vision=True,  # Включить vision для лучшего понимания страницы
        max_actions_per_step=3,
        max_steps=50,  # Достаточно шагов для выполнения задачи
    )
    
    try:
        print("Запускаем агента...\n")
        
        # Запускаем агента
        result = await agent.run(max_steps=50)
        
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТ")
        print("=" * 80)
        
        if result:
            # result это AgentHistoryList, а не объект с history
            final_result = result.final_result()
            if final_result:
                print(f"\n{final_result}")
            
            # Показываем статистику
            print(f"\nВсего шагов: {len(result.history)}")
            print(f"Успешно: {result.is_successful()}")
            
            # Показываем последние шаги
            print("\nПоследние 3 шага:")
            for i, step in enumerate(result.history[-3:], 1):
                if step.model_output and step.model_output.current_state:
                    state = step.model_output.current_state
                    print(f"\n--- Шаг {i} ---")
                    if state.memory:
                        print(f"Memory: {state.memory[:100]}...")
                    if state.next_goal:
                        print(f"Next Goal: {state.next_goal[:100]}...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        print("Сессия сохранена в:", user_data_dir)
        print("При следующем запуске авторизация будет восстановлена")
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

if __name__ == '__main__':
    asyncio.run(test_yandex_mail())

