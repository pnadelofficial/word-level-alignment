from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch 
from docx import Document
from bs4 import BeautifulSoup
import re
import random
from typing import List, Tuple, NamedTuple, Set
from dataclasses import dataclass
from tqdm import tqdm
import openai
import anthropic
from enum import Enum

PROMPT = """
# Word Alignment Task
You are a helpful AI assistant uniquely skilled in aligning words between two languages. You will be given a sentence in one language and a sentence in another language. You will be asked to align the words in the two sentences. Multiple words in Language 1 may be aligned to single words in Language 2, but multiple words in Language 2 may NOT be aligned to multiple words in Language 1.
Simply return your alignment with no other comments or explanations. If you are unable to align a word, please leave it unaligned. If you are unsure about a word, please leave it unaligned.

## Example
{example}

## New text to align
### Language 1
{lang1}
### Language 2
{lang2}
### Alignment
""".strip() 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

@dataclass
class Alignment:
    english: str
    greek: str
    start_idx: int
    end_idx: int

class MatchType(Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    WRONG = "wrong"
    UNMATCHED_GOLD = "unmatched_gold"

@dataclass
class AlignmentMatch:
    pred_alignment: Alignment
    gold_alignment: Alignment | None
    match_type: MatchType
    overlap_score: float = 0.0
    notes: str = ""

class AlignmentScore(NamedTuple):
    complete: int
    partial: int
    wrong: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    weighted_accuracy: float
    total_alignments: int
    total_gold: int
    detailed_matches: List[AlignmentMatch] 
    unmatched_gold: List[Alignment]

class Evaluator:
    def __init__(self) -> None:
        pass

    def parse_alignment(self, text: str) -> List[Alignment]:
        alignments = []
        current_pos = 0
        
        # Split by spaces but keep brackets together
        tokens = re.findall(r'\S+(?:\[[^\]]*\])?', text)
        
        for token in tokens:
            match = re.match(r'(.*?)\[(.*?)\]', token)
            if match:
                english, greek = match.groups()
                # Handle empty alignments marked with 0
                if greek == '0':
                    continue
                alignments.append(Alignment(
                    english=english.strip('.,?!'),
                    greek=greek.strip('.,?!'),
                    start_idx=current_pos,
                    end_idx=current_pos + len(english)
                ))
            current_pos += len(token) + 1  # +1 for space
            
        return alignments

    def calculate_overlap(self, align1: Alignment, align2: Alignment) -> float:
        """Calculate the overlap between two alignments."""
        start = max(align1.start_idx, align2.start_idx)
        end = min(align1.end_idx, align2.end_idx)
        if start >= end:
            return 0.0
        
        overlap_length = end - start
        total_length = max(align1.end_idx, align2.end_idx) - min(align1.start_idx, align2.start_idx)
        return overlap_length / total_length

    def evaluate_alignments(self, pred_text: str, gold_text: str) -> AlignmentScore:
        """Evaluate predicted alignments against gold standard."""
        pred_aligns = self.parse_alignment(pred_text)
        gold_aligns = self.parse_alignment(gold_text)
        
        complete = 0
        partial = 0
        wrong = 0

        detailed_matches: List[AlignmentMatch] = []
        matched_gold: Set[int] = set()

        # Track matched gold alignments to avoid double-counting
        matched_gold = set()
        
        for pred in pred_aligns:
            best_match = None
            best_score = 0
            best_gold_idx = None
            
            for i, gold in enumerate(gold_aligns):
                if i in matched_gold:
                    continue
                    
                # Check for exact match
                if (pred.english == gold.english and pred.greek == gold.greek):
                    complete += 1
                    matched_gold.add(i)
                    best_match = None
                    detailed_matches.append(AlignmentMatch(
                        pred_alignment=pred,
                        gold_alignment=gold,
                        match_type=MatchType.COMPLETE,
                        overlap_score=1.0,
                        notes="Exact match"
                        ))
                    break
                    
                # Check for partial match
                overlap = self.calculate_overlap(pred, gold)
                if overlap > 0:
                    greek_match = ((pred.greek in gold.greek) or (gold.greek in pred.greek))
                    if greek_match and (overlap > best_score):
                        best_score = overlap
                        best_match = gold
                        best_gold_idx = i
            
            if best_match is not None:
                partial += 1
                matched_gold.add(best_gold_idx)
                notes = []
                if pred.english != best_match.english:
                    notes.append(f"English text differs: '{pred.english}' vs '{best_match.english}'")
                if pred.greek != best_match.greek:
                    notes.append(f"Greek text differs: '{pred.greek}' vs '{best_match.greek}'")

                detailed_matches.append(AlignmentMatch(
                    pred_alignment=pred,
                    gold_alignment=best_match,
                    match_type=MatchType.PARTIAL,
                    overlap_score=best_score,
                    notes="; ".join(notes)
                ))
            
            elif best_match is None and complete == 0:
                wrong += 1
                detailed_matches.append(AlignmentMatch(
                    pred_alignment=pred,
                    gold_alignment=None,
                    match_type=MatchType.WRONG,
                    notes="No match found"
                ))

        unmatched_gold = [
            gold for i, gold in enumerate(gold_aligns)
            if i not in matched_gold
        ]

        for gold in unmatched_gold:
            detailed_matches.append(AlignmentMatch(
                pred_alignment=None,
                gold_alignment=gold,
                match_type=MatchType.UNMATCHED_GOLD,
                notes="Gold standard alignment not found in prediction"
            ))

        total_pred = len(pred_aligns)
        total_gold = len(gold_aligns)
        
        precision = (complete + 0.5 * partial) / total_pred if total_pred > 0 else 0
        recall = (complete + 0.5 * partial) / total_gold if total_gold > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = complete / total_gold if total_gold > 0 else 0
        weighted_accuracy = (complete + 0.5 * partial) / total_gold if total_gold > 0 else 0

        return AlignmentScore(
            complete=complete,
            partial=partial,
            wrong=wrong,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            weighted_accuracy=weighted_accuracy,
            total_alignments=total_pred,
            total_gold=total_gold,
            detailed_matches=detailed_matches,
            unmatched_gold=unmatched_gold
        )

    def format_score(score: AlignmentScore) -> str:
        """Format alignment scores as a readable string."""
        return f"""Alignment Evaluation Results:  
    Complete matches: {score.complete}
    Partial matches: {score.partial}
    Wrong matches: {score.wrong}
    Precision: {score.precision:.3f}
    Recall: {score.recall:.3f}
    F1 Score: {score.f1:.3f}"""


class WordLevelAligner:
    def __init__(self,
                 model_path: str,
                 parrish_path:str="./Aligned Odyssey 5.docx",
                 xml_path:str="tlg0012.tlg002.perseus-grc2.xml",
                 default_oai_model='gpt-4o',
                 default_anthropic_model='claude-3-5-sonnet-20241022'
                 ) -> None:
        self.model_path = model_path
        if self.model_path == 'openai':
            print("Using OpenAI API. Make sure to use os.environ['OPENAI_API_KEY'] to set your API key.")
            self.client = openai.Client()
        elif self.model_path == 'anthropic':
            print("Using Anthropic API. Make sure to use os.environ['ANTHROPIC_API_KEY'] to set your API key.")
            self.client = anthropic.Anthropic()
        elif isinstance(self.model_path, tuple):
            self.model = model_path[0]
            self.tokenizer = model_path[1]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")

        self.parrish_path = parrish_path
        self.xml_path = xml_path
        self.doc = Document(parrish_path)
        self.xml = open(xml_path).read()
        self.default_oai_model = default_oai_model
        self.default_anthropic_model = default_anthropic_model

    def get_examples(self) -> List[Tuple[str, str, str]]:
        """
        Formats examples from the source text. Hrded coded some values to get the first 200 examples.
        Returns a list of tuples with the following structure: (Language 1, Language 2, Alignment)
        """
        first_200 = [d.text for d in self.doc.paragraphs[3:203]]
        od = BeautifulSoup(self.xml, "xml")
        line_tags = od.find('div', attrs={'n':'5'}).find_all('l')
        examples = [(re.sub(r'\s+', ' ', lt.text.strip()), re.sub('\[.*?\]','', first_200[int(lt['n'])-1]).strip(), first_200[int(lt['n'])-1].strip()) for lt in line_tags[:200]]
        return examples

    def format_example(self, example: Tuple[str, str, str]) -> str:
        """
        Puts the example into the prompt format.
        Returns a string in the format for the prompt.
        """
        formatted_example = []
        for ex in example:
            formatted_example.append(f"""
            ### Language 1\n{ex[0].strip()}\n### Language 2\n{ex[1].strip()}\n### Alignment\n{ex[2]}
            """.strip())
        return '\n'.join(formatted_example).strip()

    def fill_prompt(self, examples: List[Tuple[str, str, str]], test: Tuple[str, str, str], n_shot: int) -> Tuple[str, str]:
        """
        Fills the prompt with the examples and the test.
        Returns a tuple with the filled prompt and the correct answer.
        """
        example = self.format_example(random.sample(examples, n_shot))
        filled_prompt = re.sub('\{example\}', example, PROMPT)
        filled_prompt = re.sub('\{lang1\}', test[0], filled_prompt)
        filled_prompt = re.sub('\{lang2\}', test[1], filled_prompt)
        correct = test[2]
        return filled_prompt, correct

    def align_words(self, prompt: str, max_new_tokens:int=1024) -> str:
        """
        Aligns the words in the two languages.
        Returns the alignment.
        """
        messages = [
            {'role':'user', 'content':prompt}
        ]

        if self.model_path == 'openai':
            pred = self.client.chat.completions.create(
                model=self.default_oai_model,
                messages=messages,
            )
            pred = pred.choices[0].message.content
        elif self.model_path == 'anthropic':
            pred = self.client.messages.create(
                model=self.default_anthropic_model,
                messages=messages,
                max_tokens=max_new_tokens   
            )
            pred = pred.content[0].text
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt")# .to('cuda')
            outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = re.split("### Alignmentassistant\n\n", raw_output)[-1].strip()
        return pred
    
    def evaluate(self, to_test:int=10, n_shot:int=5, max_new_tokens:int=1024) -> List[Tuple[str, str, AlignmentScore]]:
        """
        Evaluates the model on the word alignment task.
        Returns a list of alignment scores zipped with the test examples.
        """
        examples = self.get_examples()
        
        # Should make this changeable
        test = examples[100:]
        examples = examples[:100]

        _all = [self.fill_prompt(examples, t, n_shot) for t in test[:to_test]]
        all_prompts = [a[0] for a in _all]
        all_correct = [a[1] for a in _all]
        evaluator = Evaluator()

        results = []
        for prompt in tqdm(all_prompts):
            pred = self.align_words(prompt, max_new_tokens)
            correct = all_correct.pop(0)
            score = evaluator.evaluate_alignments(pred, correct)
            results.append((pred, correct, score))
        return results
