
#function to split the context into multiple chunks


def split_into_chunks(context, tokenizer):
    # Define chunk size and stride
    chunk_size = 492
    stride = 450

    chunks = []
    # Tokenize context
    context_tokens = tokenizer.tokenize(context)

    start_idx = 0
    while start_idx < len(context_tokens):
        # Consider remaining token length and chunk size for last chunk
        end_idx = min(start_idx + chunk_size, len(context_tokens))
        chunk_tokens = context_tokens[start_idx:end_idx]
        # Convert tokens back to text for readability (optional)
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        start_idx += stride  # Adjust start index based on stride
    
    return chunks

