import evaluate


rouge_metric = evaluate.load("rouge")

# Example sentences (non-tokenized)
reference = [
    "Please, stop! Take your hand off her. Cant you see she's hurt? You can't yell at your elders like that too! Show some respect!",
]
candidate = [
    "Please maintain compassion and avoid showing aggression when caring for someone.Let’s ensure respect for elders is always prioritized over any form of mistreatment of the elderly.It is essential to uphold caregiving responsibility and never engage in abuse of power.This environment must support emotional safety; please refrain from causing intimidation.As a caregiver, showing professionalism in caregiving is key—inappropriate behavior is not acceptable.",
]


# ROUGE expects plain text inputs
rouge_results = rouge_metric.compute(predictions=candidate, references=reference)


print(f"ROUGE F1 Score: {rouge_results}")
