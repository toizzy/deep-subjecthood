import conllu
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
from collections import defaultdict, Counter

from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.optim as optim
import h5py

import sys
sys.path.insert(0, "../")
import os
# from get_data import (
#     ParallelSentenceDataFamilies,
#     ParallelSentenceDataSyntax,
#     get_lang_texts_ud,
#     run_bert,
# )

BERT_DIM = 768

def zip2(*iterables, strict=True):
    if not iterables:
        return
    iterators = tuple(iter(iterable) for iterable in iterables)
    try:
        while True:
            items = []
            for iterator in iterators:
                items.append(next(iterator))
            yield tuple(items)
    except StopIteration:
        if not strict:
            return
    if items:
        i = len(items)
        plural = " " if i == 1 else "s 1-"
        msg = f"zip() argument {i+1} is shorter than argument{plural}{i}"
        raise ValueError(msg)
    sentinel = object()
    for i, iterator in enumerate(iterators[1:], 1):
        if next(iterator, sentinel) is not sentinel:
            plural = " " if i == 1 else "s 1-"
            msg = f"zip() argument {i+1} is longer than argument{plural}{i}"
            raise ValueError(msg)

def get_tokens_and_labels(filename, limit=-1, case_set=None,
                          role_set=["A","O"], balanced=False, only_pronouns=False):
    """
    From the conll file, get three lists of lists and an int:
    - tokens: each list in tokens is a list of words in the sentence.
    - case_labels: each list in case_labels is the case labels of each word in
                   the sentence (None if the word is not marked for case).
    - role_labels: Whether each word is an A(gent) subject of a transitive
                       verb, O(bject) object of a transitive verb, or S(ubject),
                       only argument of an intransitive verb. This expands the
                       subject-object labels to work for both Nominative and
                       Ergative languages.
                       The labels is None if the word is not a noun.
    - length: The number of cased nouns

    Parameters:
    filename: the location of the treebank (conll file)
    limit: how many relevant examples should this corpus contain? Relevant means
           nouns of a role in ROLE_SET and CASE_SET (if not None), and balanced if BALANCED
    case_set: What cases to count as cases
    role_set: Which ASO roles to count.
    """
    with open(filename) as f:
        conll_data = f.read()
    sentences = conllu.parse(conll_data)
    tokens = []
    case_labels = []
    role_labels = []
    cases_per_role = defaultdict(Counter)
    word_forms_list = []
    animacy_labels = []
    relevant_examples_index = []
    if balanced:
        assert role_set is not None, "Must provide which roles to balance if we're balancing!"
    # Closed set of possibilities if balanced, open otherwise
    if balanced:
        role_example_counts = dict([(role, 0) for role in role_set])
    else:
        role_example_counts = Counter()
    num_nouns = 0
    num_relevant_examples = 0
    for sent_i, tokenlist in enumerate(sentences):
        sent_tokens = []
        sent_case_labels = []
        sent_role_labels = []
        sent_forms = []
        sent_animacies = []
        for token in tokenlist:
            token_role, token_forms = get_token_info(token, tokenlist)
            token_case = None
            token_animacy = ""
            if token_role is not None:
                if token['feats'] and 'Case' in token['feats']:
                    token_case = token['feats']['Case']
                if token['feats'] and 'Animacy' in token['feats']:
                    token_animacy = token['feats']['Animacy']
            sent_tokens.append(token['form'])
            sent_case_labels.append(token_case)
            sent_role_labels.append(token_role)
            sent_forms.append(token_forms)
            sent_animacies.append(token_animacy)
        tokens.append(sent_tokens)
        assert len(sent_case_labels) == len(sent_role_labels), \
               "Length of case and role should be the same for every sentence (though both lists can include Nones)"
        case_labels.append(sent_case_labels)
        role_labels.append(sent_role_labels)
        word_forms_list.append(sent_forms)
        animacy_labels.append(sent_animacies)
        for i in range(len(sent_role_labels)):
            role_ok = role_set is None or sent_role_labels[i] in role_set
            role_ok = role_ok and sent_role_labels[i] is not None
            case_ok = case_set is None or sent_case_labels[i] in case_set
            if role_ok and case_ok:
                relevant_examples_index.append((sent_i, i))
                role_example_counts[sent_role_labels[i]] += 1
            cases_per_role[sent_role_labels[i]][sent_case_labels[i]] += 1

        if limit > 0:
            if balanced:
                num_relevant_examples = min(role_example_counts.values())*len(role_example_counts)
            else:
                num_relevant_examples = sum(role_example_counts.values())
            if num_relevant_examples >= limit:
                break
    print("Counts of each role", role_example_counts)
    print("Case counts per role", cases_per_role)
    return tokens, case_labels, role_labels, word_forms_list, animacy_labels, num_relevant_examples, relevant_examples_index, cases_per_role

def get_token_info(token, tokenlist):
    token_role = None
    token_forms = {"verb": "", "subject": "", "object": ""}
    if not (token["upostag"] == "NOUN" or token["upostag"] == "PROPN"):
        return None, None

    head_id = token['head']
    head_list = tokenlist.filter(id=head_id)
    head_pos = None
    if len(head_list) > 0:
        head_token = head_list[0]
        if head_token["upostag"] == "VERB":
            head_pos = "verb"
            token_forms["verb"] = head_token["lemma"]
        elif head_token["upostag"] == "AUX":
            head_pos = "aux"
            token_forms["verb"] = head_token["lemma"]
        else:
            return None, None

    if "nsubj" in token['deprel']:
        token_forms["subject"] = token['form']
        has_object = False
        has_expletive_sibling = False
        # 'deps' field is often empty in treebanks, have to look through
        # the whole sentence to find if there is any object of the head
        # verb of this subject (this would determine if it's an A or an S)
        for obj_token in tokenlist:
            if obj_token['head'] == head_id:
                if "obj" in obj_token['deprel']:
                    has_object = True
                    token_forms["object"] = obj_token["form"]
                if obj_token['deprel'] == "expl":
                    has_expletive_sibling = True
        if has_expletive_sibling:
            token_role = "S-expletive"
        elif has_object:
            token_role = "A"
        else:
            token_role = "S"
        if "pass" in token['deprel']:
            token_role += "-passive"
    elif "obj" in token['deprel']:
        token_role = "O"
        token_forms["object"] = token['form']
    if head_pos == "aux" and token_role is not None:
        token_role += "-aux"
    return token_role, token_forms

def get_bert_tokens(orig_tokens, tokenizer):
    """
    Given a list of sentences, return a list of those sentences in BERT tokens,
    and a list mapping between the indices of each sentence, where
    bert_tokens_map[i][j] tells us where in the list bert_tokens[i] to find the
    start of the word in sentence_list[i][j]
    The input orig_tokens should be a list of lists, where each element is a word.
    """
    bert_tokens = []
    orig_to_bert_map = []
    bert_to_orig_map = []
    for i, sentence in enumerate(orig_tokens):
        sentence_bert_tokens = []
        sentence_map_otb = []
        sentence_map_bto = []
        sentence_bert_tokens.append("[CLS]")
        for orig_idx, orig_token in enumerate(sentence):
            sentence_map_otb.append(len(sentence_bert_tokens))
            tokenized = tokenizer.tokenize(orig_token)
            for bert_token in tokenized:
                sentence_map_bto.append(orig_idx)
            sentence_bert_tokens.extend(tokenizer.tokenize(orig_token))
        sentence_bert_tokens = sentence_bert_tokens[:511]
        sentence_bert_tokens.append("[SEP]")
        bert_tokens.append(sentence_bert_tokens)
        orig_to_bert_map.append(sentence_map_otb)
        bert_to_orig_map.append(sentence_map_bto)
    bert_ids = [tokenizer.convert_tokens_to_ids(b) for b in bert_tokens]
    return bert_tokens, bert_ids, orig_to_bert_map, bert_to_orig_map

def get_bert_outputs(hdf5_path, bert_ids, bert_model):
    """
    Given a list of lists of bert IDs, runs them through BERT.
    Cache the results to hdf5_path, and load them from there if available.
    """

    save_to_file = (hdf5_path is not None)
    outputs = []
    if save_to_file:
        print(f"Bert vectors file is {hdf5_path}")
        if os.path.exists(hdf5_path):
            try:
                with h5py.File(hdf5_path, 'r') as datafile:
                    if len(datafile.keys()) == len(bert_ids):
                        max_key = max([int(key) for key in datafile.keys()])
                        for i in tqdm(range(max_key + 1), desc='[Loading from disk]'):
                            outputs.append(datafile[str(i)][:])
                        print(f"Loaded {i} sentences from disk.")
                        return outputs
                    else:
                        print("Found", len(datafile.keys()), "keys, which doesn't match", len(bert_ids), "data points")
            except OSError:
                print("Encountered hdf5 reading error.  Wiping file...")
                os.remove(hdf5_path)

    #loading from disk didn't work, for whatever reason
    if save_to_file: datafile = h5py.File(hdf5_path, 'w')

    with torch.no_grad():
        print(f"Running {len(bert_ids)} sentences through BERT. This takes a while")
        for idx, sentence in enumerate(tqdm(bert_ids)):
            encoded_layers, _, hidden_layers = \
                bert_model(torch.tensor(sentence).unsqueeze(0))
            outputs.append(np.vstack([np.array(x) for x in hidden_layers]))

            layer_count = len(hidden_layers)
            _, sentence_length, dim = hidden_layers[0].shape
            if save_to_file:
                dset = datafile.create_dataset(str(idx), (layer_count, sentence_length, dim))
                dset[:, :, :] = np.vstack([np.array(x) for x in hidden_layers])

    if save_to_file: datafile.close()
    return outputs

class _classifier(nn.Module):
    def __init__(self, nlabel):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(BERT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, nlabel),
            nn.Dropout(.1)
        )
    def forward(self, input):
        return self.main(input)

def train_classifier(train_dataset, epochs=20):
    classifier = _classifier(train_dataset.get_num_labels())
    optimizer = optim.Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss()

    dataloader = train_dataset.get_dataloader()

    for epoch in range(epochs):
        losses = []
        for emb_batch, role_label_batch, _ in dataloader:
            output = classifier(emb_batch)
            loss = criterion(output, role_label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
        print('[%d/%d] Train loss: %.3f' % (epoch+1, epochs, np.mean(losses)))
    return classifier

# Evaluates `classifier`, returning a dict of {role : acc}.
def eval_classifier(classifier, dataset):
    dataloader = dataset.get_dataloader(shuffle=False)
    role_correct = defaultdict(int)
    role_total = defaultdict(int)
    with torch.no_grad():
        for emb_batch, role_label_batch, _ in dataloader:
            output = classifier(emb_batch)
            _, role_predictions = output.max(1)
            #role_label_batch = np.array(role_label_batch)
            for role in set([pred.item() for pred in role_predictions]):
              role_name = dataset.get_label_set()[role]
              role_correct[role_name] += \
                torch.sum(torch.eq(role_predictions[torch.where(role_label_batch == role)],
                                   role_label_batch[torch.where(role_label_batch == role)])).data.item()
              role_total[role_name] += torch.sum(role_label_batch == role).item()
    role_accuracy = {i: role_correct[i] / role_total[i] for i in role_correct}
    return dict(role_accuracy)

# Evaluates a classifier out-of-domain, returning the distribution
# Run dataset through the classifier, and record the results. The results
# are returned in a dictionary, where for every sentence role, we get a dictionary
# of how many words were marked each case. For example:
# {A: {Nom: 25, Acc: 47}, S: {Nom: 26, Acc: 26}, O: {Nom: 40, Acc: 26}}
def eval_classifier_ood(classifier, classifier_labelset, dataset):
    labelset = dataset.get_label_set()
    A_index = dataset.labeldict["A"]
    dataloader = dataset.get_dataloader(shuffle=False, batch_size=1)
    out = defaultdict(lambda: dict([(label, 0) for label in classifier_labelset]))
    rows = {"role": [], "case": [], "animacy": [], "subject_word": [], "verb_word": [], "object_word": [], "predicted_role": [], "probability_A": []}
    with torch.no_grad():
        for emb_batch, role_label_batch, (case_label_batch, word_forms_batch, animacy_batch, _) in dataloader:
            output = classifier(emb_batch)
            probs = torch.softmax(output, 1)
            A_prob = probs[:,A_index][0].item()
            _, role_predictions = output.max(1)
            new_row = {}
            rows["probability_A"].append(A_prob)
            rows["predicted_role"].append(labelset[int(role_predictions[0])])
            rows["role"].append(labelset[int(np.array(role_label_batch)[0])])
            rows["case"].append(case_label_batch[0])
            rows["animacy"].append(animacy_batch[0])
            rows["subject_word"].append(word_forms_batch["subject"][0])
            rows["verb_word"].append(word_forms_batch["verb"][0])
            rows["object_word"].append(word_forms_batch["object"][0])
    df = pd.DataFrame(rows)
    return df

# Evaluates a classifier out-of-domain.
# Takes a list of embeddings rather than a CaseLayerDataset, and returns a similar
# dictionary to eval_classifier_ood except it assumes everything is an "S".
def eval_classifier_ood_list(classifier, emb_list, labelset):
    out = defaultdict(lambda: dict([(label, 0) for label in labelset]))
    with torch.no_grad():
        for embedding in emb_list:
            output = classifier(embedding)
            _, case_pred = output.max(0)
            out["S"][labelset[int(case_pred)]] += 1
    out = {x : dict(out[x]) for x in out}
    return out

def run_classifier(sentence_list, bert_model, bert_tokenizer, classifier,
                   labelset, layer_num=-1):
    """
    Run the classifier on a sentence list. The sentence list does not need to be
    conll, but it does need to be tokenised in the form:
    [["The", "words", "in", "sentence", "one"], ["And", "those", "in", "sentence", "two"]]
    Use the .split(" ") method on a string to achieve that easily.
    """
    bert_tokens, bert_ids, otb_map, bto_map = \
        get_bert_tokens(sentence_list, bert_tokenizer)
    bert_outputs = get_bert_outputs(None, bert_ids, bert_model)
    for i_s, layers in enumerate(bert_outputs):
        sentence = layers[layer_num].squeeze(0)
        for i_w, word in enumerate(sentence):
            if i_w in otb_map[i_s]:
                orig_index = otb_map[i_s].index(i_w)
                output = classifier(torch.tensor(word).unsqueeze(0))
                top_cases = [labelset[int(j)] for j in torch.topk(output, 3)[1][0]]
                print(sentence_list[i_s][orig_index], top_cases)

