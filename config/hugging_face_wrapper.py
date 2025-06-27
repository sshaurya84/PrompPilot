class HuggingFaceLLMWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _call(self, prompt, stop=None):
        output = self.pipeline(prompt)
        return output[0]['generated_text']
