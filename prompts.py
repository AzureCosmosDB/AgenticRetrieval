"""Prompt templates used by the Decomposed RAG pipeline."""

PRELIMINARY_PROMPT = """You are a helpful assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
1. ONLY use information explicitly stated in the context below
2. DO NOT make assumptions or infer information not directly stated
3. DO NOT use any external knowledge
4. If the context does not contain enough information to fully answer the question, clearly state what information IS available and what information IS MISSING
5. Be precise and cite specific details from the context
6. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Question: {question}

Context Documents:
{context}

Provide your answer in the following format:

ANSWER FROM CONTEXT:
[Your answer based strictly on the provided context]

INFORMATION GAPS:
[List any aspects of the question that cannot be answered from the provided context]

Response:"""

EFFICIENT_PRELIMINARY_PROMPT = """You are a helpful assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
1. ONLY use information explicitly stated in the context below
2. DO NOT make assumptions or infer information not directly stated
3. DO NOT use any external knowledge
4. If the context does not contain enough information to fully answer the question, clearly state what information IS available and what information IS MISSING
5. Be precise and cite specific details from the context
6. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Question: {question}

Context Documents:
{context}

Provide your answer in the following format:

CONCISE ANSWER FROM CONTEXT:
[Your concise answer based strictly on the provided context]

INFORMATION GAPS:
[List any aspects of the question that cannot be answered from the provided context]

Response:"""

SUBQUESTION_PROMPT = """You are a helpful assistant that answers questions STRICTLY based on the provided context.

IMPORTANT RULES:
1. ONLY use information explicitly stated in the context below
2. DO NOT make assumptions or infer information not directly stated
3. DO NOT use any external knowledge
4. If the context does not contain enough information to fully answer the question, clearly state what information IS available and what information IS MISSING
5. Be precise and cite specific details from the context
6. Provide a COMPREHENSIVE and DETAILED answer - include all relevant information from the context
7. Extract and include specific values, numbers, specifications, and technical details when available
8. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Question: {question}

Context Documents:
{context}

Provide a VERBOSE and COMPREHENSIVE answer that includes ALL relevant information from the context.
Include specific details, values, and technical specifications where available.

ANSWER FROM CONTEXT:
[Your detailed answer based strictly on the provided context - be thorough and include all relevant details]

INFORMATION GAPS:
[List any aspects of the question that cannot be answered from the provided context]

Response:"""

REGENERATE_PROMPT = """You are a helpful assistant that synthesizes information to provide a comprehensive answer.

You have:
1. An original question
2. A previous preliminary answer (which may have gaps)
3. Additional information from follow-up sub-questions and their answers

Your task is to generate an UPDATED and MORE COMPLETE answer by incorporating the new information from the sub-questions.

IMPORTANT RULES:
1. ONLY use information from the previous answer and the sub-question answers provided
2. DO NOT make assumptions or add external knowledge
3. Integrate the new information smoothly into a coherent answer
4. If gaps still remain, clearly identify them
5. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Original Question: {question}

Previous Preliminary Answer:
{previous_answer}

Additional Information from Sub-questions:
{sub_qa_context}

Provide your updated answer in the following format:

ANSWER FROM CONTEXT:
[Your updated answer incorporating all available information]

INFORMATION GAPS:
[List any aspects of the question that still cannot be answered]

Response:"""

GAP_DECOMPOSE_PROMPT = """You are a helpful assistant that identifies what additional information is needed to fully answer a question.

Given:
1. An original question
2. A preliminary answer based on initial context (which may be incomplete)
3. The information gaps identified in that answer

Generate sub-questions that will help fill these gaps and provide a complete answer.

Guidelines:
- Focus on the INFORMATION GAPS identified in the preliminary answer
- Each sub-question should be SIMPLE and cover only ONE specific aspect
- Each sub-question should target a single missing piece of information
- Do NOT combine multiple aspects into one sub-question
- Sub-questions should be self-contained and answerable independently
- Keep sub-questions short and focused
- Maximum {max_sub_questions} sub-questions

Original Question: {question}

Preliminary Answer:
{preliminary_answer}

Return your response as a JSON array of strings.
Example: ["What is the maximum temperature?", "What is the minimum temperature?"]

Sub-questions to fill gaps:"""

SYNTHESIS_PROMPT = """You are a helpful assistant that synthesizes information to answer questions comprehensively.

Original Question: {original_question}

Preliminary Answer (from initial retrieval):
{preliminary_answer}

Sub-questions and their answers:
{sub_qa_pairs}

Based on the above information, provide a comprehensive answer to the original question.
Synthesize the information coherently, avoid repetition, and ensure the answer directly addresses the original question.

Prioritize information from the sub-question answers that fill gaps in the preliminary answer.

At the end, add a summary that directly answers the question in a concise way. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Format your response as:
[Your comprehensive answer here]

SUMMARY: [Direct answer to the question]

Final Answer:"""

EFFICIENT_REGENERATE_PROMPT = """You are a helpful assistant that synthesizes information to provide a comprehensive answer.

You have:
1. An original question
2. A previous answer (which may have gaps)
3. New context documents retrieved to address the gaps

Your task is to generate an UPDATED and MORE COMPLETE but concise answer by incorporating the new information from the context documents.

IMPORTANT RULES:
1. ONLY use information from the previous answer and the new context documents provided
2. DO NOT make assumptions or add external knowledge
3. Integrate the new information smoothly into a coherent answer
4. If gaps still remain, clearly identify them
5. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible

Original Question: {question}

Previous Answer:
{previous_answer}

New Context Documents (retrieved for identified information gaps):
{context}

Provide your updated answer in the following format:

CONCISE ANSWER FROM CONTEXT:
[Your updated concise answer incorporating all available information]

INFORMATION GAPS:
[List any aspects of the question that still cannot be answered]

Response:"""

EFFICIENT_SYNTHESIS_PROMPT = """You are a helpful assistant that answers questions.

Original Question: {original_question}

Preliminary Answer (from initial retrieval):
{preliminary_answer}

Answers from successive retrieval rounds (each round retrieved additional context to fill gaps):
{round_answers}

Based on the above information, provide an answer to the question in a concise way. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible.

Concise answer:"""
