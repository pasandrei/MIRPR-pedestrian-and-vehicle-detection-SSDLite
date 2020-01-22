from jaad_data import inference


def dummy_input(model, inp, handler):
    inference.feed_to_model(model, inp, handler)
