CFG_JUDGE = """
You are an expert at evaluating answers to open-ended questions about fine-grained video understanding and reasoning.
You will evaluate the model's response to the given question based on the complete fine-grained description of the video ("VideoCaption") and the human-written correct answer ("CorrectAnswer"), and score the model's performance on two dimensions: "correctness" and "detailedness", each within the range of 0 to 10.

Use the VideoCaption as the **primary factual reference**. The CorrectAnswer is only an auxiliary reference; if VideoCaption and CorrectAnswer conflict, always follow the VideoCaption.

The open-ended task types include:
  1) Counterfactual Interaction ("CIA"): The question asks how an action that did **not** occur would be executed. Check whether the answer points out the error in the question and explains the correct way of execution.
  2) Counterfactual Relationship ("CRS"): The question describes an incorrect temporal or causal relationship. Check whether the answer identifies the error and provides the correct sequence and outcome.
  3) Functional Intention ("FI"): Focus on the intention behind a specific action.
  4) Global Intention ("GI"): Focus on the overall purpose or goal of the protagonist’s sequence of actions.
  5) Counterfactual Intention ("CIT"): Focus on how execution should change if the protagonist had a different intention.
  6) Process Monitoring ("PM"): Focus on the stage, smoothness, or blockage of task execution in relation to the global intention.
  7) Strategy Evaluation ("SE"): Focus on the appropriateness, efficiency, and coordination complexity of the chosen execution strategy.
  8) Counterfactual Evaluation ("CEU"): Focus on how changing a specific execution detail would alter the quality, efficiency, or success of the action.

### Evaluation Process

#### Phase 1: Zero-Tolerance Check (for CIA / CRS)

If the TaskType is CIA or CRS, the CandidateAnswer must explicitly reject OR implicitly correct the false premise in the question stem. 

– If the model explicitly identifies that the question’s description is wrong or inconsistent with the video, OR the model implicitly corrects the premise by accurately describing the true events in the video that contradict the false assumption, proceed to Phase 2. 

– If the model passively accepts the false premise as true and hallucinates based on it (i.e., acquiescence), assign Correctness=0 and Detailedness=0, and skip directly to the final output.

If the TaskType is not CIA or CRS, skip Phase 1 and proceed directly to Phase 2.

#### Phase 2: Scoring on Two Dimensions

### Dimension 1: Correctness (0–10 points)

Evaluate whether the CandidateAnswer is factually consistent with the VideoCaption and **truly answers the reasoning demand of the Question**.

**Correctness Rating Criteria:**

**9–10 points (Perfect)**
- Completely accurate. Matches the VideoCaption facts and the core logic of the CorrectAnswer.
-  For counterfactual questions, accurately corrects the factual error in the premise (either explicitly through refutation or implicitly through factual description) and provides the true execution details based on the video.
- No critical errors about who did what, when, where, why, or with what.

**7–8 points (High)**
- Correct main conclusion and key reasoning steps.
- May have minor omissions or small inconsistencies compared to the CorrectAnswer, but no clear contradictions with the VideoCaption.
- Correctly addresses the intended reasoning type (e.g., intention, process, strategy, counterfactual).

**5–6 points (Medium)**
- Partially correct: captures the general idea or part of the reasoning chain, but misses important steps or conditions.
- May include one notable mistake or several minor slips, or mix correct and incorrect causal/temporal links.
- Still shows some meaningful alignment with the Question and VideoCaption.

**3–4 points (Low)**
- Significant errors: misinterprets the core of the Question or misreads key facts from the VideoCaption.
- Multiple incorrect claims about actions, actors, intentions, or temporal/causal relations.
- Reasoning type alignment is weak: the answer often talks about the wrong aspect (e.g., static appearance instead of intention).

**1–2 points (Poor)**
- Mostly irrelevant to the Question or heavily hallucinates facts not supported by the VideoCaption.
- Offers almost no correct reasoning about the video.

**0 points (Fail)**
- Fails the Phase 1 Check for CIA/CRS (does not identify the false premise), or the response is pure gibberish and cannot be meaningfully evaluated.

### Dimension 2: Detailedness (0–10 points)

Evaluate whether the CandidateAnswer is **comprehensive, precise, and focused** when addressing the required reasoning type, including conditions, nuances, and evidence from the VideoCaption.

**Detailedness Rating Criteria:**

**9–10 points (Rich)**
- Thoroughly explains the relevant “why” and/or “how” for the given reasoning type (intention, process, strategy, counterfactual, etc.).
- Uses multiple specific details from the VideoCaption (objects, movements, sequence of steps, outcomes, constraints).
- Comparable in depth and nuance to the CorrectAnswer.

**7–8 points (Good)**
- Covers the main reasoning steps and most key conditions.
- Includes several concrete details from the VideoCaption, though may miss some secondary nuances.
- Explanation is coherent and informative, but not fully exhaustive.

**5–6 points (Average)**
- Provides a generally correct explanation but remains high-level or somewhat generic.
- Mentions some relevant elements (e.g., main intention or main cause–effect link) but lacks fine-grained detail or specific evidence.
- Limited use of explicit cues from the VideoCaption.

**3–4 points (Weak)**
- Brief or superficial; focuses on only one part of the reasoning while omitting other important aspects.
- Contains very few concrete details from the VideoCaption.
- Does not fully engage with the required reasoning type (e.g., just states the result without explaining why).

**1–2 points (Poor)**
- Extremely short, vague, or empty.
- Almost no usable reasoning or concrete detail.

**0 points (Fail)**
- Fails the Phase 1 Check for CIA/CRS (thus Detailedness must also be 0), or is pure gibberish with no coherent content.

## Additional Policies

- **Expression tolerance:** Paraphrasing or surface wording differences are acceptable if the meaning is equivalent.  
- **Multiple valid answers:** Different correct phrasings expressing the same reasoning are equally valid.  
- **No reward for verbosity:** Long but off-topic or repetitive text does **not** increase the Detailedness score. Only relevant, grounded details count.  
- **Correctness first:** If an answer is seriously incorrect, it should not receive high scores even if it is long or detailed.  
- **Omissions vs. errors:** Omitting some content is less severe than stating it incorrectly. Explicit errors should penalize Correctness more than simple omissions.

### Uncertain or Incomplete Evidence Policy

Even if the VideoCaption does not explicitly state the reasoning or outcome, you must **infer the most plausible interpretation** from the available evidence and context.
- Always provide a reasoned judgment rather than defaulting to pure uncertainty.
- Use **lower scores** to reflect weak or incomplete evidence instead of skipping evaluation.
- If evidence is truly minimal, assign approximately Correctness = 4–5 and Detailedness = 3–4, and clearly explain in the output that the judgment is based on limited cues.
- Avoid outputs like "Underspecified"; always justify your stance with textual evidence or reasonable inference.

## Context
VideoCaption:
{caption}
Question:
{question}
CorrectAnswer:
{correct_answer}
TaskType:
{task_type}
CandidateAnswer:
{result}

## Output (STRICT JSON)
Return **only**:
{
  "evidence_spans": ["<short quotes from VideoCaption that support your judgment>"],
  "correctness_reasoning": "<explanation of your correctness score>",
  "detailedness_reasoning": "<explanation of your detailedness score>",
  "correctness": <integer 0-10>,
  "detailedness": <integer 0-10>,
}
"""