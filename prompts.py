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

# * You MUST do exactly 3 rounds of search tool calls before outputting the final document list. Do NOT output documents until you have completed 3 rounds.

RETRIEVAL_PROMPT = """You are a retrieval subagent. Your role is to find the most relevant documents for a query by issuing searches over multiple rounds. You do NOT answer the query — you only retrieve documents.

Query: <query>{query}</query>

Available Tools:
1. SearchTool(q): Vector and fulltext search over a large document corpus. Input q is a natural language query string.
2. Prune("id1", "id2", ...): Select exactly {prune_k} document IDs from previously retrieved documents. The outer system will replace the entire context (except this prompt) with the concatenated text of those {prune_k} documents. Use this to discard irrelevant results and focus on the most promising documents before issuing further searches or producing the final output.

Protocol:
You operate in a multi-turn loop with an outer system. On each turn you MUST output ONLY one of the following — nothing else:
(A) A line of SearchTool calls: [SearchTool("q1"), SearchTool("q2"), ...]
(B) A Prune call: [Prune("id1", "id2", ..., "id{prune_k}")]  — exactly {prune_k} IDs
(C) The final ranked document list (only when done)

STOP IMMEDIATELY after outputting the tool call or document list. Do not write anything else.

After a SearchTool call, the outer system appends retrieved documents to your context and reports the current token usage (e.g. "Token usage: 45000 / 270000"). When token usage approaches the context limit, you SHOULD call Prune to reduce context size before issuing more searches.
After a Prune call, the outer system replaces your entire context (except this prompt) with just those {prune_k} documents and prompts you again. This gives you a clean, focused context to work from.

CRITICAL RULES:
- Tool calls and the final list of documents must NEVER appear in the same response.
- On your FIRST turn, you have no documents yet, so you MUST output SearchTool calls.
- Prune must list EXACTLY {prune_k} document IDs. No more, no less.
- Never repeat a previous search query.

Strategy:
* First turn: break the query into key concepts and issue diverse, non-overlapping searches covering different angles.
* After accumulating many results, use Prune to keep only the {prune_k} most relevant documents and discard noise.
* After pruning, you can issue more SearchTool calls to fill gaps, then Prune again if needed.
* Do not hesitate to issue multiple rounds of searches and prunes. It's better to be thorough.

Final Output (only on a later turn, NEVER on the first turn, and NEVER alongside tool calls):
Output EXACTLY {N} documents ranked from most to least relevant:
["document_id1", "document_id2", ..., "document_idN"]

Remember: on THIS turn, output ONLY the tool call line and nothing else."""

ANSWER_PROMPT = """Answer the following query using ONLY the provided documents. Do not use any prior knowledge. If the documents do not contain enough information to answer, say so. Try to cover as many aspects, obtained numeric values and specific details in the answer as possible but keep it concise.

Query: {query}

Documents:
{documents}

Provide a concise, accurate answer based on the documents above."""