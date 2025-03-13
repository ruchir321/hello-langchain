from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

def generate_names(demeanor: str, role: str, job: str):
    model = OllamaLLM(model="llama3.2", temperature=0.5)

    system_template = "You are a {demeanor} {role}"
    user_input_template = "Give me 5 callsigns for {job}s"

    prompt_template = ChatPromptTemplate(
        [
            ('system', system_template),
            ('user', user_input_template)
        ]
    )

    prompt = prompt_template.invoke(
        {
            "demeanor": demeanor,
            "role": role,
            "job": job
        }
    )

    return model.invoke(input=prompt)

if __name__ == "__main__":
    print(generate_names(demeanor="witty", role="wing commander", job="air transport pilot"))