import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


load_dotenv()


deployment = os.getenv("DEPLOYMENT_NAME")



llm = AzureChatOpenAI(
    azure_deployment=deployment,
    
    temperature=0  
)

three_sentence_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n\n{text}"
)


three_sentence_chain = LLMChain(
    llm=llm,
    prompt=three_sentence_prompt
)

text_about_ai = """
Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are designed to think, learn, and adapt like humans. 
These systems can perform tasks such as recognizing speech, making decisions, and identifying patterns in data. 
AI is typically categorized into two main types: narrow AI, which is specialized for a particular task such as language translation or facial recognition, 
and general AI, which would be capable of performing a wide range of tasks at the level of human intelligence. 
Recent advancements in machine learning, especially deep learning, have accelerated the growth of AI, 
enabling breakthroughs in fields such as healthcare, autonomous vehicles, finance, and robotics. 
For example, AI can assist doctors in diagnosing diseases more accurately, power self-driving cars, 
and detect fraudulent transactions in real time. Despite these benefits, AI also poses significant challenges, 
including ethical concerns, job displacement, and the potential misuse of intelligent systems. 
Researchers and policymakers are actively working to ensure that AI development is safe, fair, and beneficial to society. 
The future of AI holds immense potential, but it requires careful regulation and responsible innovation to maximize benefits while minimizing risks.
"""


print("=== 3 Sentence Summary ===")
print(three_sentence_chain.run(text_about_ai))



one_sentence_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 1 sentence:\n\n{text}"
)

one_sentence_chain = LLMChain(
    llm=llm,
    prompt=one_sentence_prompt
)

print("\n=== 1 Sentence Summary ===")
print(one_sentence_chain.run(text_about_ai))
