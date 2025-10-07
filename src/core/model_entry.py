# model_entry.py
ACRONYMS = {"lstm": "LSTM", "cnn": "CNN", "rnn": "RNN"}

class ModelEntry:
    def __init__(self, internal_name, compile_fn):
        # internal_name is a normalized ID name for the model
        self.internal_name = internal_name  # e.g., lstm
        # pretty_name is a display name for plots and logs
        self.pretty_name = self._format_pretty_name(internal_name)  # e.g., AttentionLSTM
        # compile_fn is a the associated function to compile the model
        self.compile_fn = compile_fn  

    def _format_pretty_name(self, name):
        # Convert the internal name to a pretty name
        acronyms = {"lstm": "LSTM", "cnn": "CNN", "rnn": "RNN"}
        return "".join(acronyms.get(part, part.capitalize()) for part in name.split("_"))

    def build(self, input_shape):
        return self.compile_fn(input_shape)
    
    def __str__(self):
        return self.pretty_name