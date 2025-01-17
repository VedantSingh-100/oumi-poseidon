"""This module provides access to various judge configurations for the Oumi project.

The judges are used to evaluate the quality of AI-generated responses based on
different criteria such as helpfulness, honesty, and safety.
"""

from oumi.judges.judge_court import (
    oumi_v1_xml_claude_sonnet_judge,
    oumi_v1_xml_gpt4o_judge,
    oumi_v1_xml_local_judge,
)

__all__ = [
    "oumi_v1_xml_claude_sonnet_judge",
    "oumi_v1_xml_gpt4o_judge",
    "oumi_v1_xml_local_judge",
]
