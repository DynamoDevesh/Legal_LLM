def build_augmented_prompt(user_query, context_sections, top_k=5):
    prompt = (
        "You are a legal assistant.\n"
        "Use the following legal sections to answer the user's legal query.\n\n"
        f"User Query: {user_query.strip()}\n\n"
        f"Top {top_k} Relevant Legal Sections:\n"
    )

    for section in context_sections[:top_k]:
        desc = section.get("description", "").strip()
        title = section.get("title", "").strip()
        content = desc if desc else title
        if content:
            prompt += f"- {content}\n"

    prompt += "\nExplain the legal provision in clear terms as it applies to the user's question."
    return prompt
