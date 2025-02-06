# NLP Components

## Backwards Compatibility

The current implementation maintains compatibility with older code while providing enhanced functionality. Here's how the compatibility is maintained:

### Tokenizer
- `rag_tokenizer.tokenize()` - Returns space-separated tokens when called from old code
- `rag_tokenizer.fine_grained_tokenize()` - Maintains old behavior with enhanced processing

### Query
- `query.FulltextQueryer` - Wrapper class that maintains old interface
- `hybrid_similarity()` - Maintains old signature while using new processing
- `token_similarity()` - Compatible with both string and token inputs

### Usage in Old Code