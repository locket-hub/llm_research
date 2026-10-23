import json
from pathlib import Path


input_path = Path("data/processed/candidates.jsonl")
output_path = Path("data/processed/preferences.jsonl")

max_prompts = 20


with input_path.open("r", encoding="utf-8") as infile, \
     output_path.open("w", encoding="utf-8") as outfile:

    for count, line in enumerate(infile):

        if count >= max_prompts:
            break

        record = json.loads(line)

        print("\n" + "=" * 80)
        print(f"PROMPT {record['id']}")
        print("=" * 80)
        print(record["prompt"])

        print("\nCOMPLETIONS:\n")

        for i, candidate in enumerate(record["candidates"]):
            print(f"[{i}]")
            print(candidate)
            print("-" * 80)

        print(
            "\nRank from BEST to WORST."
            "\nExample: 2 0 3 1"
        )

        while True:
            ranking = input("> ").strip().split()

            try:
                ranking = [int(x) for x in ranking]

                if sorted(ranking) == [0, 1, 2, 3]:
                    break

            except ValueError:
                pass

            print("Enter each number 0, 1, 2, 3 exactly once.")

        preference = {
            "id": record["id"],
            "prompt": record["prompt"],
            "candidates": record["candidates"],
            "ranking": ranking,
        }

        outfile.write(json.dumps(preference) + "\n")
        outfile.flush()

print(f"\nSaved preferences to {output_path}")