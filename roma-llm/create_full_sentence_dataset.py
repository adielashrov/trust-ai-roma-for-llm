import os
import random
import string
from datasets import load_dataset

# Load the SST-2 dataset
dataset = load_dataset("sst2")
test_set = dataset["test"]

# Create a folder to save the modified sentences
output_folder = "full_modified_sentences"
os.makedirs(output_folder, exist_ok=True)

def perturb_sentence(sentence,idx):
	file_path = os.path.join(output_folder, f"sentence_{idx}.txt")
	with open(file_path, 'w', encoding='utf-8') as f:
		f.write(f"Original: {original_sentence}\n\n")

		alphabet = string.ascii_letters


		for i, char in enumerate(sentence):
			if char in alphabet:
				for replacement in alphabet:
					if replacement != char:
						perturbed = sentence[:i] + replacement + sentence[i+1:]
						#print(perturbed)
						f.write(f"Modified {i+1}: {perturbed}\n")
					
		f.close()
                    


# Example usage






def modify_sentence(sentence):
    # Convert the sentence to a list of characters
    chars = list(sentence)
    
    # Choose a random position (excluding spaces)
    valid_positions = [i for i, char in enumerate(chars) if char.isalpha()]
    if not valid_positions:
        return sentence  # Return original sentence if no valid positions
    
    position = random.choice(valid_positions)
    
    # Choose a random letter (different from the original)
    original_letter = chars[position].lower()
    new_letter = random.choice([c for c in string.ascii_lowercase if c != original_letter])
    
    # Replace the letter, maintaining case
    if chars[position].isupper():
        chars[position] = new_letter.upper()
    else:
        chars[position] = new_letter
    
    # Return the modified sentence
    return ''.join(chars)

# Process each sentence in the test set
for idx, sample in enumerate(test_set):
    original_sentence = sample['sentence']
    perturb_sentence(original_sentence,idx)
    
    

    
    print(f"Processed sentence {idx+1}/{len(test_set)}")

print("All sentences processed and saved.")
