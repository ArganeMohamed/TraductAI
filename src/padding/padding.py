def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence = sequence + [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]

    return sequence

if __name__ == "__main__":
    example = [1,2,3,4,5,6,7]

    padded = pad_sequence(example, 5)

    print(example)
    print(padded)