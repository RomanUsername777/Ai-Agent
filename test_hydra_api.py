"""
Простой тест подключения к HydraAI API
"""
import asyncio
import os
from dotenv import load_dotenv

from agent import ChatOpenAI
from agent.llm.messages import UserMessage

load_dotenv()

async def test_hydra_api():
    """Тест подключения к HydraAI API"""
    
    openai_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_URL')
    
    # Если указан OPENAI_API_URL с полным путём /chat/completions, убираем его
    if base_url and '/chat/completions' in base_url:
        base_url = base_url.replace('/chat/completions', '')
    
    print(f"🔑 API Key: {openai_key[:20]}...")
    print(f"🌐 Base URL: {base_url}")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY не найден в .env")
        return
    
    # Создаём клиент OpenAI с кастомным base_url
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=openai_key,
        base_url=base_url
    )
    
    print("\n📤 Отправляем тестовый запрос...")
    
    try:
        response = await llm.ainvoke([
            UserMessage(content="Привет! Ответь одним предложением: как дела?")
        ])
        
        print(f"\n✅ Успешно получен ответ:")
        print(f"📝 {response.completion}")
        if response.usage:
            print(f"📊 Использовано токенов: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
        
    except Exception as e:
        print(f"\n❌ Ошибка при запросе:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test_hydra_api())

