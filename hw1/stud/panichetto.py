def __iter__(self):
    """
    Overwrites the __iter__() method of the superclass (torch.utils.data.IterableDataset).
    """
    sentence_pairs = self.data_json[0]
    for spair in sentence_pairs:
        if merge:  #RABARBARO
            sentence = self.merge_pair(spair=spair, sep_token=sep_token)
        #else:
        #    sentence  = spair["sentence1"]
        #    sentence2 = spair["sentence2"]
        #    sentence.extend(sentence2)
            
        len_sentence = len(sentence)

        for input_idx in range(len_sentence):
            current_word = sentence[input_idx]
            # must be a word in the vocabulary
            if current_word in self.word2id and self.keep_word(current_word):
                # left and right window indices
                min_idx = max(0, input_idx - self.window_size)
                max_idx = min(len_sentence, input_idx + self.window_size)

                window_idxs = [x for x in range(min_idx, max_idx) if x != input_idx]
                for target_idx in window_idxs:
                    # must be a word in the vocabulary
                    if sentence[target_idx] in self.word2id:
                        # index of target word in vocab
                        target = self.word2id[sentence[target_idx]]
                        # index of input word
                        current_word_id = self.word2id[current_word]
                        output_dict = {'targets':target, 'inputs':current_word_id}

                        yield output_dict

def keep_word(self, word: str):
    '''
    Implements negative sampling and returns true if we can keep the occurrence as training instance.
    '''
    z = self.frequency[word] / self.tot_occurrences
    p_keep = np.sqrt(z / 10e-3) + 1
    p_keep *= 10e-3 / z # higher for less frequent instances
    return np.random.rand() < p_keep # toss a coin and compare it to p_keep to keep the word