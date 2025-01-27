
def classifier(speed):
    if speed < 64.0:
        intent = 1
    else:
        intent = 2
    return intent
