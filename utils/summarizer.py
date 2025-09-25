from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
def summarizer_chain(input_text,llm):
    three_sentence_prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in exactly 3 sentences:\n\n{text}"
    )


    three_sentence_chain = LLMChain(
        llm=llm,
        prompt=three_sentence_prompt
    )

    print("=== 3 Sentence Summary ===")
    print(three_sentence_chain.run(input_text))



    one_sentence_prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in exactly 1 sentence:\n\n{text}"
    )

    one_sentence_chain = LLMChain(
        llm=llm,
        prompt=one_sentence_prompt
    )

    print("\n=== 1 Sentence Summary ===")
    print(one_sentence_chain.run(input_text))
