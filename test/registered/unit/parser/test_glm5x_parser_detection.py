"""Auto-detection must pick a tool parser that can actually read the template.

GLM-5.2 keeps the enable_thinking toggle; GLM-5.3 and GLM-5.3-Flash dropped it
and reason unconditionally, and Flash builds the tool call with Jinja `~`.
Without a rule for that the 5.3 templates fall out of the GLM family and land on
glm45, whose regex requires a newline after the function name — which those
templates do not emit, so tool calls stop being recognised at all.

The fixtures below are structural: they carry the markers detection reads, not a
copy of the shipped templates.
"""

import re
import unittest

from sglang.srt.parser.template_detection import (
    detect_reasoning_parser,
    detect_reasoning_pattern,
    detect_tool_call_parser,
)

GLM_VOCAB = {
    "<tool_call>", "</tool_call>", "<arg_key>", "<arg_value>",
    "<|user|>", "<|endoftext|>", "<think>", "</think>", "<|observation|>",
}


class _Tokenizer:
    def __init__(self, vocab):
        self._vocab = {token: index for index, token in enumerate(sorted(vocab))}

    def get_vocab(self):
        return self._vocab


def _template(*, toggle: bool, concat: str) -> str:
    # The toggle has to be written the way the shipped GLM-5.2 template writes
    # it, otherwise the reasoning-mode rule does not fire and the fixture tests
    # nothing.
    thinking = (
        "{% if not enable_thinking is defined %}"
        "{% set enable_thinking = true %}{% endif %}"
        if toggle
        else ""
    )
    return (
        "[gMASK]<sop>\n"
        + thinking
        + "{%- for m in messages %}<|user|>{{ m.content }}\n"
        "<think>{{ m.reasoning }}</think>\n"
        "{%- for tc in m.tool_calls %}\n"
        "{{- '<tool_call>' " + concat + " tc.name -}}\n"
        "{% for k, v in tc.arguments.items() %}<arg_key>{{ k }}</arg_key>"
        "<arg_value>{{ v }}</arg_value>{% endfor %}</tool_call>\n"
        "{%- endfor %}{%- endfor %}"
    )


# The three shipped lines, by the two properties detection actually reads.
CASES = {
    "glm-5.2": _template(toggle=True, concat="+"),
    "glm-5.3": _template(toggle=False, concat="+"),
    "glm-5.3-flash": _template(toggle=False, concat="~"),
}

# glm45 requires a newline between the function name and the first argument;
# glm47 does not. Both GLM lines render without one.
GLM45_DETAIL = re.compile(r"<tool_call>(.*?)(?:\\n|\n)(.*)</tool_call>", re.DOTALL)
GLM47_DETAIL = re.compile(r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL)
SHIPPED_CALL = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>"


class TestGlm5xParserDetection(unittest.TestCase):
    def test_every_glm5x_line_selects_the_glm_parsers(self):
        for name, template in CASES.items():
            with self.subTest(template=name):
                force, config = detect_reasoning_pattern(template)
                tokenizer = _Tokenizer(GLM_VOCAB)
                self.assertEqual(
                    detect_tool_call_parser(template, tokenizer, config, force), "glm47"
                )
                self.assertEqual(
                    detect_reasoning_parser(template, tokenizer, config, force), "glm45"
                )

    def test_selected_parser_can_read_the_rendered_call(self):
        """The chosen parser has to survive the format, not just be selected."""
        matched = GLM47_DETAIL.search(SHIPPED_CALL)
        self.assertIsNotNone(matched)
        self.assertEqual(matched.group(1), "get_weather")
        # The rejected alternative is rejected for a reason: it needs a newline.
        self.assertIsNone(GLM45_DETAIL.search(SHIPPED_CALL))

    def test_non_glm_template_is_left_alone(self):
        template = "{%- for m in messages %}<|im_start|>{{ m.content }}{%- endfor %}"
        force, config = detect_reasoning_pattern(template)
        tokenizer = _Tokenizer({"<|im_start|>"})
        self.assertNotEqual(
            detect_tool_call_parser(template, tokenizer, config, force), "glm47"
        )


if __name__ == "__main__":
    unittest.main()
