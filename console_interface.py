"""Интерактивный консольный интерфейс для агента"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from agent import Agent, Browser, ChatOpenAI, ChatAnthropic
from agent.browser import BrowserProfile

load_dotenv()


class ConsoleInterface:
    """Интерактивный консольный интерфейс для работы с агентом"""
    
    def __init__(self, session_name: str = None, headless: bool = False):
        """
        Инициализация интерфейса
        
        Args:
            session_name: Имя сессии для сохранения (по умолчанию: "default")
            headless: Запуск браузера в headless режиме (по умолчанию: False - браузер видимый)
        """
        self.session_name = session_name or "default"
        self.headless = headless  # По умолчанию False - браузер всегда видимый
        self.agent: Optional[Agent] = None
        self.browser: Optional[Browser] = None
        self.llm = None
        self.running = False
        self.task_history = []
        
        # Определяем файл для сохранения сессии
        self.storage_state_path = Path(f'./{self.session_name}_storage_state.json')
    
    def _init_agent(self):
        """Инициализация агента и браузера"""
        if self.agent is not None:
            return
        
        # Выбор LLM провайдера
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        anthropic_model = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929')
        
        if openai_key:
            base_url = os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_URL')
            # Если указан OPENAI_API_URL с полным путём /chat/completions, убираем его
            if base_url and '/chat/completions' in base_url:
                base_url = base_url.replace('/chat/completions', '')
            # Если используется HydraAI или другой нестандартный провайдер, отключаем response_format
            # так как он может не поддерживаться, и добавляем схему в системный промпт вместо этого
            is_hydra = base_url and 'hydraai.ru' in base_url.lower()
            
            # Определяем, является ли модель Claude (для HydraAI)
            is_claude = any(claude_name in openai_model.lower() for claude_name in ['claude', 'sonnet', 'haiku', 'opus'])
            
            print(f"🤖 Используем OpenAI-совместимый API (модель: {openai_model})")
            if is_hydra and is_claude:
                print(f"   ⚠️  Используется Claude через HydraAI - убедитесь, что модель {openai_model} доступна")
            
            self.llm = ChatOpenAI(
                model=openai_model, 
                api_key=openai_key, 
                base_url=base_url,
                dont_force_structured_output=is_hydra,
                add_schema_to_system_prompt=is_hydra  # Добавляем схему в промпт для HydraAI
            )
        elif anthropic_key:
            print(f"🤖 Используем Anthropic (модель: {anthropic_model})")
            self.llm = ChatAnthropic(model=anthropic_model)
        else:
            raise ValueError("❌ Необходим OPENAI_API_KEY или ANTHROPIC_API_KEY в .env файле")
        
        # Создаем браузер
        from agent.browser.profile import ViewportSize
        # Всегда передаем путь к storage_state, даже если файл еще не существует
        # StorageStateWatchdog создаст файл при первом сохранении
        browser_profile = BrowserProfile(
            storage_state=str(self.storage_state_path),
            user_data_dir=None,
        )
        
        self.browser = Browser(
            headless=self.headless,
            browser_profile=browser_profile,
            window_size=ViewportSize(width=1200, height=700),  # Размер окна браузера
        )
        
        # Функция для запроса ввода от пользователя (для капчи)
        def user_input_prompt(prompt: str) -> str:
            print(f'\n🔒 {prompt}')
            print('Введите "готово" (или "done") когда закончите:', end=' ')
            return input()
        
        # Создаем агента
        self.agent = Agent(
            task="",  # Задача будет задаваться позже
            llm=self.llm,
            browser=self.browser,
            use_vision=True,
            max_actions_per_step=3,
            user_input_callback=user_input_prompt,
        )
        
        print(f"✅ Агент инициализирован")
        print(f"💾 Сессия будет сохранена в: {self.storage_state_path}")
    
    def print_help(self):
        """Вывод справки"""
        help_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 АГЕНТ ДЛЯ АВТОМАТИЗАЦИИ БРАУЗЕРА

Команды:
  help, h         - Показать эту справку
  history, hist   - История выполненных задач
  clear           - Очистить историю
  tabs, t         - Показать открытые вкладки браузера
  exit, quit, q   - Выход из программы

Использование:
  Введите задачу на естественном языке:
  - "Перейди на [сайт]"
  - "Прочитай последние письма и удали спам"
  - "Найди вакансии и откликнись на них"
  - "Закажи еду из ресторана"

Примеры:
  > Перейди на нужный сайт
  > Найди кнопку входа и нажми на неё
  > Заполни форму и отправь её
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        print(help_text)
    
    def print_history(self):
        """Вывод истории задач"""
        if not self.task_history:
            print("📜 История пуста\n")
            return
        
        print(f"\n📜 История задач ({len(self.task_history)}):")
        print("━" * 80)
        for i, entry in enumerate(self.task_history[-10:], 1):  # Последние 10 задач
            status = "✅" if entry.get("success") else "❌"
            task = entry.get("task", "N/A")
            steps = entry.get("steps", 0)
            result = entry.get("result", "")
            
            print(f"{i}. {status} {task}")
            print(f"   Шагов: {steps}")
            if result:
                print(f"   Результат: {result[:100]}...")
            print()
    
    async def _print_tabs_async(self):
        """Вывод информации об открытых вкладках (асинхронная версия)"""
        if not self.browser:
            print("🌐 Браузер не инициализирован\n")
            return
        
        try:
            # Получаем информацию о вкладках через метод get_tabs
            if hasattr(self.browser, 'get_tabs'):
                tabs = await self.browser.get_tabs()
                if not tabs:
                    print("📑 Нет открытых вкладок\n")
                    return
                
                print(f"\n📑 Открытые вкладки ({len(tabs)}):")
                print("━" * 80)
                for i, tab in enumerate(tabs, 1):
                    marker = "👉 " if i == 1 else "   "  # Первая вкладка - текущая
                    title = tab.title if hasattr(tab, 'title') else 'Без названия'
                    url = tab.url if hasattr(tab, 'url') else 'about:blank'
                    tab_id = tab.target_id[-4:] if hasattr(tab, 'target_id') else '????'
                    print(f"{marker}[{i}] {title} (ID: {tab_id})")
                    print(f"    {url}")
                print()
            else:
                print("📑 Информация о вкладках недоступна\n")
        except Exception as e:
            print(f"⚠️  Ошибка получения информации о вкладках: {e}\n")
    
    def print_tabs(self):
        """Вывод информации об открытых вкладках"""
        if not self.browser:
            print("🌐 Браузер не инициализирован\n")
            return
        asyncio.run(self._print_tabs_async())
    
    async def execute_task(self, task: str):
        """Выполнение задачи"""
        if not self.agent:
            self._init_agent()
        
        print(f"\n🚀 Выполняю задачу: {task}\n")
        print("━" * 80)
        
        try:
            # Обновляем задачу агента
            self.agent.task = task
            # Обновляем задачу в MessageManager, чтобы она передавалась в промпт
            if self.agent._message_manager:
                self.agent._message_manager.task = task
            
            # Запускаем агента
            result = await self.agent.run(max_steps=50)
            
            # Сохраняем в историю
            history_entry = {
                "task": task,
                "success": result.is_successful() if result else False,
                "steps": len(result.history) if result and result.history else 0,
                "result": result.final_result() if result else "",
            }
            self.task_history.append(history_entry)
            
            # Сохраняем сессию после выполнения задачи
            if self.browser and hasattr(self.browser, 'browser_session') and self.browser.browser_session:
                try:
                    await self.browser.browser_session.export_storage_state(self.storage_state_path)
                    logger.debug(f'💾 Сессия сохранена в: {self.storage_state_path}')
                except Exception as e:
                    logger.warning(f'⚠️  Ошибка при сохранении сессии: {e}')
            
            # Выводим результат
            print("\n" + "━" * 80)
            if result and result.history:
                final_result = result.final_result()
                if final_result:
                    print(f"\n✅ Результат:\n{final_result}\n")
                else:
                    print(f"\n✅ Задача выполнена\n")
                
                print(f"📊 Статистика:")
                print(f"   • Шагов выполнено: {len(result.history)}")
                print(f"   • Успешно: {'Да' if result.is_successful() else 'Нет'}")
            else:
                print("\n⚠️  Задача не завершена\n")
            
            print("━" * 80)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Прервано пользователем\n")
        except Exception as e:
            print(f"\n❌ Ошибка при выполнении задачи: {e}\n")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Запуск интерактивного интерфейса"""
        self.running = True
        
        print("\n" + "=" * 80)
        print("🤖 АГЕНТ ДЛЯ АВТОМАТИЗАЦИИ БРАУЗЕРА")
        print("=" * 80)
        print("\n💡 Введите 'help' для справки или задачу для выполнения\n")
        
        # Инициализируем агента при первом запуске
        try:
            self._init_agent()
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}\n")
            return
        
        while self.running:
            try:
                user_input = input("> ").strip()
                if not user_input:
                    continue
                
                command = user_input.lower()
                
                if command in ['exit', 'quit', 'q']:
                    break
                elif command in ['help', 'h']:
                    self.print_help()
                    continue
                elif command in ['tabs', 't']:
                    self.print_tabs()
                    continue
                elif command in ['history', 'hist']:
                    self.print_history()
                    continue
                elif command in ['clear']:
                    self.task_history = []
                    print("🧹 История очищена\n")
                    continue
                
                # Выполняем задачу
                asyncio.run(self.execute_task(user_input))
            
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Выход из программы...\n")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}\n")
        
        # Закрываем браузер и сохраняем сессию
        print("\n💾 Сохраняю сессию...")
        if self.browser:
            try:
                # Сохраняем storage_state перед закрытием
                if hasattr(self.browser, 'browser_session') and self.browser.browser_session:
                    storage_state = asyncio.run(self.browser.browser_session.export_storage_state(self.storage_state_path))
                    print(f"✅ Сессия сохранена в: {self.storage_state_path}")
                asyncio.run(self.browser.close())
            except Exception as e:
                print(f"⚠️  Ошибка при сохранении сессии: {e}")
        print("👋 До свидания!\n")

