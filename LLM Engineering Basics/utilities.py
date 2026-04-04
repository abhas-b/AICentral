# Token counts are on response.usage (returned by the API)
def usage_metrics(response):
    usage = response.usage
    print(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)