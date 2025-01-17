import tempfile
from pathlib import Path
from typing import Final

import jsonlines

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import (
    Conversation,
    Message,
    MessageContentItem,
    Role,
    Type,
)
from oumi.inference import NativeTextInferenceEngine
from oumi.utils.image_utils import load_image_png_bytes_from_path
from oumi.utils.io_utils import get_oumi_root_directory
from tests.markers import requires_cuda_initialized

TEST_IMAGE_DIR: Final[Path] = (
    get_oumi_root_directory().parent.parent.resolve() / "tests" / "testdata" / "images"
)


def _get_default_text_model_params() -> ModelParams:
    return ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
        chat_template="gpt2",
        tokenizer_pad_token="<|endoftext|>",
    )


def _get_default_image_model_params() -> ModelParams:
    return ModelParams(
        model_name="llava-hf/llava-1.5-7b-hf",
        model_max_length=1024,
        trust_remote_code=True,
        chat_template="llava",
    )


def _get_default_inference_config() -> InferenceConfig:
    return InferenceConfig(
        generation=GenerationParams(max_new_tokens=5, temperature=0.0, seed=42)
    )


def _setup_input_conversations(filepath: str, conversations: list[Conversation]):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).touch()
    with jsonlines.open(filepath, mode="w") as writer:
        for conversation in conversations:
            json_obj = conversation.to_dict()
            writer.write(json_obj)
    # Add some empty lines into the file
    with open(filepath, "a") as f:
        f.write("\n\n\n")


#
# Tests
#
def test_infer_online():
    engine = NativeTextInferenceEngine(_get_default_text_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
            Message(
                content="Hello again!",
                role=Role.USER,
            ),
        ],
        metadata={"foo": "bar"},
        conversation_id="123",
    )
    expected_result = [
        Conversation(
            messages=[
                *conversation.messages,
                Message(
                    content="The first time I saw",
                    role=Role.ASSISTANT,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
    ]
    result = engine.infer_online([conversation], _get_default_inference_config())
    assert expected_result == result


def test_infer_online_empty():
    engine = NativeTextInferenceEngine(_get_default_text_model_params())
    result = engine.infer_online([], _get_default_inference_config())
    assert [] == result


def test_infer_online_to_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    content="Touche!",
                    role=Role.USER,
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        expected_result = [
            Conversation(
                messages=[
                    *conversation_1.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            ),
            Conversation(
                messages=[
                    *conversation_2.messages,
                    Message(
                        content="The U.S.",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = _get_default_inference_config()
        inference_config.output_path = str(output_path)
        result = engine.infer_online(
            [conversation_1, conversation_2],
            inference_config,
        )
        assert result == expected_result
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations


def test_infer_from_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        conversation = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation])
        expected_result = [
            Conversation(
                messages=[
                    *conversation.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            )
        ]
        result = engine.infer_from_file(
            str(input_path),
            _get_default_inference_config(),
        )
        assert expected_result == result
        inference_config = _get_default_inference_config()
        inference_config.input_path = str(input_path)
        infer_result = engine.infer(inference_config=inference_config)
        assert expected_result == infer_result


def test_infer_from_file_empty():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [])
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        inference_config = _get_default_inference_config()
        result = engine.infer_from_file(str(input_path), inference_config)
        assert [] == result
        inference_config.input_path = str(input_path)
        infer_result = engine.infer(inference_config=inference_config)
        assert [] == infer_result


def test_infer_from_file_to_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    content="Touche!",
                    role=Role.USER,
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation_1, conversation_2])
        expected_result = [
            Conversation(
                messages=[
                    *conversation_1.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            ),
            Conversation(
                messages=[
                    *conversation_2.messages,
                    Message(
                        content="The U.S.",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = _get_default_inference_config()
        inference_config.output_path = str(output_path)
        result = engine.infer_online(
            [conversation_1, conversation_2],
            inference_config,
        )
        assert result == expected_result
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations


@requires_cuda_initialized()
def test_infer_from_file_to_file_with_images():
    png_image_bytes_great_wave = load_image_png_bytes_from_path(
        TEST_IMAGE_DIR / "the_great_wave_off_kanagawa.jpg"
    )
    png_image_bytes_logo = load_image_png_bytes_from_path(
        TEST_IMAGE_DIR / "oumi_logo_dark.png"
    )

    test_prompt: str = "Generate a short, descriptive caption for this image!"

    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_image_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    type=Type.COMPOUND,
                    content=[
                        MessageContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=png_image_bytes_great_wave,
                        ),
                        MessageContentItem(
                            type=Type.TEXT,
                            content=test_prompt,
                        ),
                    ],
                )
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    type=Type.COMPOUND,
                    content=[
                        MessageContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=png_image_bytes_logo,
                        ),
                        MessageContentItem(
                            type=Type.TEXT,
                            content=test_prompt,
                        ),
                    ],
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation_1, conversation_2])
        expected_result = [
            Conversation(
                messages=[
                    *conversation_1.messages,
                    Message(
                        content="2 boats are in the",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            ),
            Conversation(
                messages=[
                    *conversation_2.messages,
                    Message(
                        content="4 white squares make up",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = _get_default_inference_config()
        inference_config.output_path = str(output_path)

        result = engine.infer_online(
            [conversation_1, conversation_2],
            inference_config,
        )
        assert result == expected_result
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations
