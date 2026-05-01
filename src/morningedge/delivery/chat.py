"""RAG-powered chat over MorningEdge data.

Retrieves relevant articles + narratives + scores via DuckDB, hands them
to Gemini with a citation-anchored prompt, and returns answers with source
links. Lets users ask things like 'why is private credit sentiment negative
this week?' and get a real, grounded answer.
"""
