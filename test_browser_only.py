"""
Тест только браузера и загрузки сессии (без LLM)
Проверяет что браузер открывается и сессия загружается
"""
import asyncio
from pathlib import Path
import json

from agent import Browser

async def test_browser_session():
    """Тест загрузки сессии браузера"""
    
    print("=" * 80)
    print("ТЕСТ: Загрузка сессии браузера")
    print("=" * 80)
    
    # Проверяем наличие storage state
    storage_state_path = Path('./yandex_mail_storage.json')
    if storage_state_path.exists():
        print(f"\n✅ Найден storage state: {storage_state_path}")
        with open(storage_state_path) as f:
            storage_data = json.load(f)
            cookies_count = len(storage_data.get('cookies', []))
            print(f"   Cookies в сессии: {cookies_count}")
    else:
        print(f"\n⚠️  Storage state не найден: {storage_state_path}")
    
    user_data_dir = Path('./yandex_mail_session')
    
    # Создаем браузер
    print(f"\nСоздаем браузер с user_data_dir: {user_data_dir}")
    browser = Browser(
        headless=False,  # Видимый браузер
        user_data_dir=str(user_data_dir),
    )
    
    try:
        print("\nЗапускаем браузер...")
        await browser.start()
        print("✅ Браузер запущен!")
        
        # Получаем текущую страницу
        current_url = await browser.get_current_page_url()
        print(f"✅ Текущий URL: {current_url}")
        
        # Проверяем cookies
        cookies = await browser.cookies()
        print(f"✅ Cookies в браузере: {len(cookies)}")
        
        if cookies:
            print("\nПервые 3 cookies:")
            for cookie in cookies[:3]:
                print(f"  - {cookie.get('name', 'N/A')}: {cookie.get('domain', 'N/A')}")
        
        # Пробуем перейти на mail.yandex.ru
        print("\nПереходим на mail.yandex.ru...")
        await browser.navigate('https://mail.yandex.ru')
        
        await asyncio.sleep(2)  # Ждем загрузки
        
        current_url = await browser.get_current_page_url()
        print(f"✅ Текущий URL после перехода: {current_url}")
        
        # Получаем состояние страницы
        state = await browser.get_browser_state_summary()
        print(f"✅ Состояние страницы получено")
        print(f"   Title: {state.title[:50] if state.title else 'N/A'}...")
        print(f"   Интерактивных элементов: {len(state.dom_state.selector_map) if state.dom_state else 0}")
        
        print("\n" + "=" * 80)
        print("✅ ТЕСТ УСПЕШЕН!")
        print("=" * 80)
        print("\nБраузер работает, сессия загружается.")
        print("Для полного теста с агентом нужен API ключ (OPENAI_API_KEY или ANTHROPIC_API_KEY)")
        
        # Держим браузер открытым 5 секунд для визуальной проверки
        print("\nБраузер останется открытым 5 секунд для проверки...")
        await asyncio.sleep(5)
        
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
    asyncio.run(test_browser_session())

