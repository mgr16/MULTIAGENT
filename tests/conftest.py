import pytest

class Dummy:
    def __init__(self, text):
        self.text = text

@pytest.fixture(autouse=True)
def stub_llm_calls(monkeypatch):
    """
    Stubs para que los tests sean offline:
    - call_llm devuelve JSONs mínimos válidos según el agente.
    - call_llm_mm (visión) devuelve estructura vacía.
    """
    from app import agents
    import app.models.openai_llm as ollm

    def fake_call_llm(model, messages, json_object=False, temperature=0.2, max_tokens=None):
        sys_plus_user = "\n".join([m.get("content","") for m in messages])
        if "routing expert" in sys_plus_user.lower():
            return ('{"domain":"general","confidence":0.9,"suggested_agents":["planner"]}', 
                    ollm.LLMUsage(model=model, input_tokens=10, output_tokens=10, cost_usd=0.0))
        if "design multi-agent plans" in sys_plus_user.lower():
            return ('{"steps":[{"name":"gather","agents":["rag","web_search"]},{"name":"analyze","agents":["data"]},{"name":"draft","agents":["summary"]},{"name":"critique","agents":["critic"]},{"name":"finalize","agents":["summary"]}],"stop_condition":"final_answer"}',
                    ollm.LLMUsage(model=model, input_tokens=10, output_tokens=20, cost_usd=0.0))
        if "careful quantitative analyst" in sys_plus_user.lower():
            return ('{"analysis":"ok","key_numbers":[{"name":"x","value":1}],"assumptions":[]}',
                    ollm.LLMUsage(model=model, input_tokens=10, output_tokens=30, cost_usd=0.0))
        if "produce a precise, sourced answer" in sys_plus_user.lower():
            return ('{"final_answer":"respuesta sintetizada","citations":["local"]}',
                    ollm.LLMUsage(model=model, input_tokens=10, output_tokens=30, cost_usd=0.0))
        if "adversarial critic" in sys_plus_user.lower():
            return ('{"confidence":0.8,"conflicts":0,"issues":[]}',
                    ollm.LLMUsage(model=model, input_tokens=10, output_tokens=30, cost_usd=0.0))
        # fallback genérico JSON
        return ('{"ok":true}', ollm.LLMUsage(model=model, input_tokens=5, output_tokens=5, cost_usd=0.0))

    def fake_call_llm_mm(model, parts, json_object=True, temperature=0.2, max_tokens=800):
        return ('{"contains_chart":false,"contains_text":false,"any_numbers":false}', 
                ollm.LLMUsage(model=model, input_tokens=5, output_tokens=5, cost_usd=0.0))

    monkeypatch.setattr(ollm, "call_llm", fake_call_llm)
    monkeypatch.setattr(ollm, "call_llm_mm", fake_call_llm_mm)
