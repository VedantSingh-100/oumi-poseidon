import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from tqdm.auto import tqdm
from typing_extensions import Self

from oumi.core.configs import JudgeConfig
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role, TemplatedMessage
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    RemoteInferenceEngine,
)


class BaseJudgeOutput(ABC, TemplatedMessage):
    raw_judgement: Optional[str] = None

    @classmethod
    def from_xml_output(cls, raw_judgement: Optional[str]) -> Optional[Self]:
        """Parses the judgement from XML-like tags in the raw output.

        Args:
            raw_judgement: The raw judgement string to parse.

        Returns:
            Optional[Self]: An instance of the class with parsed attributes,
                or None if parsing fails.
        """
        if not raw_judgement:
            return None

        attributes = {}
        # Regex pattern to match XML-like tags and their content
        # Captures the tag name in group 1 and the content between tags in group 2
        # For example, "<label>True</label>" would match as ("label", "True")
        pattern = r"<(\w+)>(.*?)</\1>"
        matches = re.findall(pattern, raw_judgement, re.DOTALL)

        for attr_name, attr_value in matches:
            attributes[attr_name] = attr_value.strip()

        return cls(**attributes, raw_judgement=raw_judgement)

    @classmethod
    def from_json_output(cls, raw_judgement: Optional[str]) -> Optional[Self]:
        """Parses the judgement from JSON."""
        if not raw_judgement:
            return None

        try:
            judgement_data = json.loads(raw_judgement)
            return cls(**judgement_data, raw_judgement=raw_judgement)
        except json.JSONDecodeError:
            return None

    @property
    def label(self):
        """Convert the judgement to a boolean or Likert scale label.

        This method should be overridden by subclasses to provide the actual
        conversion logic.
        """
        return self.raw_judgement

    @property
    def fields(self):
        """Return the fields of the judgement."""
        fields = self.model_dump()
        fields.pop("raw_judgement", None)
        fields.pop("template", None)
        fields.pop("role", None)
        return fields


class BaseJudge(ABC):
    def __init__(
        self,
        config: JudgeConfig,
        inference_engine: Optional[BaseInferenceEngine] = None,
    ):
        """Initialize the Judge."""
        self._config = config
        self._attributes = config.attributes
        if len(config.attributes) < 1:
            raise ValueError(
                "At least one attribute must be specified in the judge configuration."
            )

        if inference_engine is None:
            self.inference_engine = self._create_inference_engine(config)
        else:
            self.inference_engine = inference_engine

    def judge(
        self,
        raw_inputs: Union[List[Conversation], List[dict], List[Message]],
    ) -> List[Dict[str, BaseJudgeOutput]]:
        """Judge the given conversations."""
        # Convert the raw user inputs into a list of JudgeInput classes
        # A JudgeInput is the unit of what needs to be judged, and could be a
        # prompt, request/answer pair or a full conversation
        judge_inputs = []

        for raw_input in raw_inputs:
            if isinstance(raw_input, dict):
                judge_input = self._transform_dict_input(raw_input)
            elif isinstance(raw_input, TemplatedMessage):
                judge_input = raw_input
            elif isinstance(raw_input, Conversation):
                judge_input = self._transform_conversation_input(raw_input)
            else:
                raise ValueError(f"Unsupported conversation type: {type(raw_input)}")

            judge_inputs.append(judge_input)

        results = {}

        for attribute_name in self._attributes.keys():
            # Generate the full judging prompt for each attribute
            # This includes the judge system prompt, and any few shot examples
            # That are included in the judge config.
            judgement_prompts = [
                self.build_judgement_prompt(judge_input, attribute_name=attribute_name)
                for judge_input in tqdm(judge_inputs)
            ]

            # Run inference for the attribute's prompt
            # We batch the inference for a single attribute together to maximally
            # benefit from kv prefix caching (system prompt, few shot examples)
            raw_judgements = self._infer(judgement_prompts)

            # Parse the raw judge output (a string) into a JudgeOutput object
            judgements = []
            for conversation in raw_judgements:
                judgement = conversation.messages[-1].content

                parsed_judgement = self._transform_model_output(judgement)

                judgements.append(
                    {
                        "raw_judgement": judgement,
                        "fields": parsed_judgement.fields,
                        "label": parsed_judgement.label,
                    }
                )

            results[attribute_name] = judgements

        # Results are now in the form
        #   {attribute: judgements for attribute in attributes}
        # Transform to
        #   [{attribute: judgement} for judgement in judgements]
        outputs = []
        for idx in range(len(raw_inputs)):
            output_dict = {}
            for attribute_name in self._attributes.keys():
                output_dict[attribute_name] = results[attribute_name][idx]
            outputs.append(output_dict)

        return outputs

    def build_judgement_prompt(
        self, judge_input: Message, attribute_name: Optional[str]
    ) -> Conversation:
        """Generate judge prompts for a dataset."""
        if attribute_name is None:
            if len(self._attributes) > 0:
                raise ValueError(
                    "attribute_name must be specified when there are multiple"
                    " attributes to judge."
                )
            else:
                # If there's only one attribute, use it
                attribute_name = next(iter(self._attributes))

        if attribute_name not in self._attributes:
            raise KeyError(
                f"Attribute '{attribute_name}' not found in config.attributes"
            )

        attribute = self._attributes[attribute_name]
        messages = attribute.messages.copy()
        messages.append(Message(content=judge_input.content, role=Role.USER))

        return Conversation(
            messages=messages,
            metadata={
                "judge_attribute_name": attribute_name,
            },
        )

    def _infer(self, conversations: List[Conversation]) -> List[Conversation]:
        """Judge a single attribute."""
        metadatas = [convo.metadata for convo in conversations]

        responses = self.inference_engine.infer(
            input=conversations, generation_config=self._config.generation
        )

        assert len(responses) == len(metadatas)

        for response, metadata in zip(responses, metadatas):
            response.metadata.update(metadata)

        return responses

    def _create_inference_engine(self, config: JudgeConfig) -> BaseInferenceEngine:
        """Create the inference engine."""
        # TODO: Initialize the appropriate inference engine based on the config
        # For now, we default to the remote inference engine
        # Users can override this method to provide their own inference engine
        # to the constructor of the Judge class.
        model_name = config.model.model_name.lower()
        if "gguf" in model_name:
            return LlamaCppInferenceEngine(config.model)
        elif "claude" in model_name:
            return AnthropicInferenceEngine(config.model)
        else:
            return RemoteInferenceEngine(config.model)

    @abstractmethod
    def _transform_conversation_input(self, conversation: Conversation) -> Message:
        raise NotImplementedError

    @abstractmethod
    def _transform_dict_input(self, raw_input: Dict[str, Any]) -> Message:
        raise NotImplementedError

    @abstractmethod
    def _transform_model_output(self, model_output) -> BaseJudgeOutput:
        raise NotImplementedError