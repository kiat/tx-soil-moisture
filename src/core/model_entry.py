# model_entry.py
ACRONYMS = {"lstm": "LSTM", "cnn": "CNN", "rnn": "RNN"}

class ModelEntry:
    def __init__(self, internal_name, compile_fn):
        self.internal_name = internal_name  # e.g., attention_lstm
        self.pretty_name = self._format_pretty_name(internal_name)  # e.g., AttentionLSTM
        self.compile_fn = compile_fn  # function that takes input_shape

    def _format_pretty_name(self, name):
        acronyms = {"lstm": "LSTM", "cnn": "CNN", "rnn": "RNN"}
        return "".join(acronyms.get(part, part.capitalize()) for part in name.split("_"))

    def build(self, input_shape):
        return self.compile_fn(input_shape)
    
    def __str__(self):
        return self.pretty_name