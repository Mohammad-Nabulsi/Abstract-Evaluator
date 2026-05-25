from json import loads
from json_repair import repair_json


class Preprocessor:
    """
    Preprocessor class for creating or applying prompts to a Hugging Face dataset.
    Also handles tokenization of the input data to prepare it for model inference or training.
    """
    def __init__(self):
        pass


    @staticmethod
    def quantify(dataset, tokenizer):
        """
        Computes the total number of tokens in the dataset without counting the special tokens.

        :param dataset: HuggingFace Dataset
        :param tokenizer: The appropriate tokenizer.
        :return: Total number of tokens.
        """
        texts  = dataset.map(lambda x: {"text": " ".join(str(v) for v in x.values())})
        counts = 0

        for example in texts["text"]:
            tokens  = tokenizer.encode(example, add_special_tokens=False)
            counts += len(tokens)

        return counts


    @staticmethod
    def formulate(example):
        """
        Creates a prompt with system, user, and assistant messages from the input example.

        :param example: dictionary containing question, reference_answer, student_answer, mark_scheme, and optionally score and rationale.
        :return: dictionary with a formatted prompt containing system, user, and assistant messages.
        """
        question         = example["question"].strip()
        reference_answer = example["reference_answer"].strip()
        student_answer   = example["student_answer"].strip()
        mark_scheme      = example["mark_scheme"].strip()

        score            = str(example.get("score", "")).strip()
        rationale        = example.get("rationale", "").strip()

        system_content   = "You are a precise grading assistant."

        user_content     = (
            "Provide both a score and a rationale by evaluating the student's answer strictly within the mark scheme range,"
            " grading based on how well it meets the question's requirements by comparing the student answer to the reference answer.\n"
            f"Question: {question}\n"
            f"Reference Answer: {reference_answer}\n"
            f"Student Answer: {student_answer}\n"
            f"Mark Scheme: {mark_scheme}")

        assistant_content = f'{{\"score\": {score}, \"rationale\": "{rationale}"}}'

        system            = {"role": "system", "content": system_content}
        user              = {"role": "user", "content": user_content}
        assistant         = {"role": "assistant", "content": assistant_content}

        prompt            = [system, user, assistant]

        return {"prompt": prompt}


    def reformat(self, dataset):
        """
        Applies prompt formatting to each example in a Hugging Face dataset.

        :param dataset: HuggingFace dataset to transform, containing fields for prompt formatting.
        :return: A dataset with each example mapped to a prompt format.
        """
        reformatted_dataset = dataset.map(self.formulate, remove_columns=dataset.column_names)

        return reformatted_dataset


    @staticmethod
    def encode(example, tokenizer, inference=False, device=None):
        """
        Encodes a prompt into tokens using the tokenizer, with optional inference mode and device setting.

        :param example: dictionary with a prompt field formatted as a list of chat messages.
        :param tokenizer: the appropriate tokenizer.
        :param inference: whether to prepare the input for inference (adds generation prompt and returns tensors).
        :param device: the device to move encoded tensors to during inference (e.g. cuda, cpu).
        :return: encoded prompt, either as raw tokens or a tensor dictionary for inference.
        """
        if device is None: device = "cpu"

        if inference:
            encoded = tokenizer.apply_chat_template(example["prompt"],
                                                    tokenize             =True,
                                                    add_generation_prompt=True,
                                                    return_tensors       ="pt",
                                                    return_dict          =True).to(device)

            return encoded

        else:
            encoded = tokenizer.apply_chat_template(example["prompt"],
                                                    tokenize             =False,
                                                    add_generation_prompt=False)

            return tokenizer(encoded, truncation=True)


    def tokenize(self, dataset, tokenizer):
        """
        Tokenizes a HuggingFace dataset using a given tokenizer for model training or inference.

        :param dataset: HuggingFace dataset to tokenize, containing prompt fields compatible with the tokenizer.
        :param tokenizer: The tokenizer to apply to the dataset.
        :return: A dataset with tokenized outputs (e.g., input_ids, attention_mask).
        """
        tokenized_dataset = dataset.map(lambda example: self.encode(example=example, tokenizer=tokenizer),
                                        batched       =True,
                                        remove_columns=dataset.column_names)

        return tokenized_dataset


class Postprocessor:
    """
    Handles post-processing of text and prompt, including extracting structured data
    and removing response sections from prompts.
    """

    def __init__(self):
        pass


    @staticmethod
    def extract(text, actual=True):
        """
        Extracts the assistant's response content and parses the json-like string into a dict if `actual` is false.

        :param text: list of chat messages or a string containing a json-like response.
        :param actual: if true, extracts the assistant's message content; if false, parses the response string as json.
        :return: extracted string content or a parsed dictionary response.
        """
        if actual:
            raw = next(item["content"] for item in text if item["role"] == "assistant")
            raw = repair_json(raw)
            return loads(s=raw)

        start = text.find("{")
        end   = text.find("}", start)

        if start == -1 or end == -1: return text

        response = text[start:end + 1].strip()
        response = repair_json(response)

        try   : return loads(s=response)
        except: return response


    @staticmethod
    def strip(prompt):
        """
        Removes assistant message from the prompt, keeping only system and user roles.

        :param prompt: list of chat messages including system, user, and assistant roles and content.
        :return: list of messages with only system and user roles.
        """
        system_user_messages = [msg for msg in prompt if msg["role"] != "assistant"]

        return system_user_messages