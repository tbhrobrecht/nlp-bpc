# evaluate_blimp.py
import argparse
import torch
import datasets
import os
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from functools import partial
from tqdm import tqdm

SPM_PATH = './data/tokenizer.model'
BLIMP_SUBSETS = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
DEVICE = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, required=True, choices=['encoder', 'decoder'])
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--batch_size', type=int, default=64)


def tokenize_encoder(examples, tokenizer):
    batch = {
        "good_inputs": [],
        "bad_inputs": [],
        "good_labels": [],
        "bad_labels": [],
    }

    for i in range(len(examples['sentence_good'])):
        good_tokens = tokenizer.encode(examples['sentence_good'][i], add_special_tokens=False)
        bad_tokens = tokenizer.encode(examples['sentence_bad'][i], add_special_tokens=False)
        batch["good_inputs"].append(good_tokens)
        batch["bad_inputs"].append(bad_tokens)
        batch["good_labels"].append(good_tokens)
        batch["bad_labels"].append(bad_tokens)

    return batch

def tokenize_decoder(examples, tokenizer):
    batch = {
        "good_inputs": [],
        "bad_inputs": [],
        "good_labels": [],
        "bad_labels": [],
    }

    for i in range(len(examples['sentence_good'])):
        good_tokens = tokenizer.encode(examples['sentence_good'][i])
        bad_tokens = tokenizer.encode(examples['sentence_bad'][i])
        batch["good_inputs"].append(good_tokens[:-1])
        batch["bad_inputs"].append(bad_tokens[:-1])
        batch["good_labels"].append(good_tokens[1:])
        batch["bad_labels"].append(bad_tokens[1:])

    return batch


def padding_collate_fn(batch, max_len=1024, left_padding=False):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}
    for key in batch[0]:
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if "labels" in key:
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            key_len = min(max_len, len(sample[key]))
            if left_padding:
                padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])
            else:
                padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch


def evaluate_decoder(model, dataloader, tokenizer):
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch in dataloader:
            good_inputs = batch['good_inputs'].to(device=DEVICE)
            bad_inputs = batch['bad_inputs'].to(device=DEVICE)
            good_labels = batch['good_labels'].to(device=DEVICE)
            bad_labels = batch['bad_labels'].to(device=DEVICE)

            good_outputs = model(good_inputs, attention_mask=good_inputs != 0).logits
            bad_outputs = model(bad_inputs, attention_mask=bad_inputs != 0).logits
            good_loss = loss_fn(good_outputs.view(-1, good_outputs.shape[-1]), good_labels.view(-1))
            bad_loss = loss_fn(bad_outputs.view(-1, bad_outputs.shape[-1]), bad_labels.view(-1))

            good_loss = good_loss.view(good_inputs.shape[0], -1).mean(dim=1)
            bad_loss = bad_loss.view(bad_inputs.shape[0], -1).mean(dim=1)
            for b in range(len(good_loss)):
                if good_loss[b] < bad_loss[b]:
                    correct += 1
                total += 1
    return correct / total


def evaluate_encoder(model, dataloader, tokenizer):
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch in dataloader:
            good_loss = 0.0
            bad_loss = 0.0

            good_inputs = batch['good_inputs'].to(device=DEVICE)
            bad_inputs = batch['bad_inputs'].to(device=DEVICE)
            good_labels = batch['good_labels'].to(device=DEVICE)
            bad_labels = batch['bad_labels'].to(device=DEVICE)

            max_len = batch['good_inputs'].shape[1]
            for i in range(max_len):
                masked_good_inputs = good_inputs.clone()
                masked_good_inputs[:, i] = tokenizer.mask_token_id
                good_outputs = model(masked_good_inputs, attention_mask=good_inputs != 0).logits
                good_loss += loss_fn(good_outputs.view(-1, good_outputs.shape[-1]), good_labels.view(-1))

            max_len = batch['bad_inputs'].shape[1]
            for i in range(max_len):
                masked_bad_inputs = bad_inputs.clone()
                masked_bad_inputs[:, i] = tokenizer.mask_token_id
                bad_outputs = model(masked_bad_inputs, attention_mask=bad_inputs != 0).logits
                bad_loss += loss_fn(bad_outputs.view(-1, bad_outputs.shape[-1]), bad_labels.view(-1))

            good_loss = good_loss.view(good_inputs.shape[0], -1).mean(dim=1)
            bad_loss = bad_loss.view(bad_inputs.shape[0], -1).mean(dim=1)

            for b in range(len(good_loss)):
                if good_loss[b] < bad_loss[b]:
                    correct += 1
                total += 1
    return correct / total

def main():
    args = parser.parse_args()
    model_name = "prajjwal1/bert-tiny" if args.model_type == "encoder" else "sshleifer/tiny-gpt2"
    evaluate_fn = evaluate_encoder if args.model_type == "encoder" else evaluate_decoder
    if not args.model_path:
        args.model_path = model_name

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenize_fn = partial(tokenize_encoder if args.model_type == "encoder" else tokenize_decoder, tokenizer=tokenizer) # need to make the function unary for map()

    # load model
    if args.model_type == "encoder":
        model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)

    results = {}

    print("Evaluating on BLIMP subsets...")
    for subset in tqdm(BLIMP_SUBSETS):
        # load dataset and tokenize
        dataset = datasets.load_dataset('json', data_files={'train': os.path.join(args.data_path, f'{subset}.jsonl')})
        dataset = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=dataset['train'].column_names) # map works with functions that return a dictionary
        dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn)
        result = evaluate_fn(model, dataloader, tokenizer)
        results[subset] = result


    for subset, result in results.items():
        print(f" -- {subset}: {result:.4f}")

    average = sum(results.values()) / len(results)
    print(f"Average: {average:.4f}")


if __name__ == '__main__':
    main()