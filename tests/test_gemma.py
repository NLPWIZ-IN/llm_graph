
import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.integrations.gemma_quant import (
    QuantizedEmbedding, replace_with_quant_layers,
)
from transformers.utils.quantization_config import GemmaQuantizationConfig

from llm_graph import constants
from llm_graph import DiagramConfig, LLMGraphError, create_visualizer
from llm_graph.errors import CATALOG
from llm_graph.inspection import ModuleInspector
from llm_graph.mermaid import escape_label
from llm_graph.schema import Edge, GraphView, Node
from llm_graph.views import ViewBuilder


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "tests/fixtures/gemma_e2b_profile.json"


class GemmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = Gemma4Config(**json.loads(PROFILE.read_text()))
        with torch.device("meta"):
            cls.model = AutoModelForCausalLM.from_config(c)
            quant = GemmaQuantizationConfig.from_dict(c.quantization_config)
            replace_with_quant_layers(cls.model, quant, quant.modules_to_not_convert)
        cls.model.config.quantization_config = quant
        cls.visualizer = create_visualizer()
        cls.report = cls.visualizer.visualize(cls.model)

    def assert_code(self, code, function):
        with self.assertRaises(LLMGraphError) as caught:
            function()
        self.assertEqual(caught.exception.code, code)
        json.dumps(caught.exception.to_dict())
        return caught.exception

    def test_local_profile_and_variants(self):
        layers = self.report.graph.layers
        self.assertEqual(len(layers), 35)
        self.assertEqual(Counter(x.attention_type for x in layers), {
            constants.ATTENTION_SLIDING: 28, constants.ATTENTION_FULL: 7,
        })
        groups = Counter(x.signature for x in layers)
        self.assertEqual(sorted(groups.values()), [3, 4, 12, 16])
        variants = ViewBuilder().build(self.report.graph, DiagramConfig(views=(constants.VIEW_VARIANTS,)))
        self.assertEqual([len(v.member_indices) for v in variants], [12, 3, 16, 4])
        self.assertEqual(variants[1].member_indices, (4, 9, 14))
        self.assertEqual(variants[3].member_indices, (19, 24, 29, 34))
        for layer in layers[15:]:
            self.assertEqual(layer.kv_source, 14 if layer.attention_type == constants.ATTENTION_FULL else 13)
            self.assertEqual(layer.intermediate_size, 12288)
        self.assertTrue(all(p.is_meta for p in self.model.parameters()))

    def test_shared_kv_graph_has_no_local_kv(self):
        graph = self.report.graph.decoders[15]
        self.assertFalse({constants.NODE_K, constants.NODE_V} & {n.id for n in graph.nodes})
        self.assertTrue(any(e.kind == constants.EDGE_SHARED_STATE for e in graph.edges))
        self.assertTrue({constants.NODE_K, constants.NODE_V} <= {n.id for n in self.report.graph.decoders[0].nodes})

    def test_packed_dimensions_and_bits(self):
        modules = {m.path: m for m in self.report.graph.modules}
        q = modules["model.language_model.layers.0.self_attn.q_proj"]
        self.assertEqual(q.properties[constants.FIELD_IN_FEATURES], 1536)
        self.assertEqual(q.properties[constants.FIELD_OUT_FEATURES], 2048)
        self.assertEqual(q.properties[constants.FIELD_NUM_BITS], 4)
        self.assertEqual(q.properties[constants.FIELD_CONFIGURED_BITS], 4)
        self.assertEqual(next(t.shape for t in q.tensors if t.name == "weight"), (2048, 768))
        later = modules["model.language_model.layers.15.mlp.gate_proj"]
        self.assertEqual(later.properties[constants.FIELD_NUM_BITS], 2)
        self.assertEqual(later.properties[constants.FIELD_CONFIGURED_BITS], 2)
        emb = modules["model.language_model.embed_tokens"]
        self.assertEqual(emb.properties[constants.FIELD_NUM_BITS], 2)
        self.assertTrue(any(t.name == "embedding_quantized" for t in emb.tensors))

    def test_inspection_does_not_dequantize_or_run(self):
        before = [(id(p), p.device, p.dtype, p._version) for p in self.model.parameters()]
        config_before = self.model.config.to_dict()
        with patch.object(QuantizedEmbedding, "_dequantize_weights", side_effect=AssertionError("dequant")):
            with patch.object(nn.Module, "_call_impl", side_effect=AssertionError(constants.METHOD_FORWARD)):
                self.visualizer.visualize(self.model)
        self.assertEqual(before, [(id(p), p.device, p.dtype, p._version) for p in self.model.parameters()])
        self.assertEqual(config_before, self.model.config.to_dict())

    def test_bad_inputs_and_backend(self):
        self.assert_code(constants.ERROR_INVALID_MODEL_INPUT, lambda: self.visualizer.visualize(None))
        self.assert_code(constants.ERROR_UNSUPPORTED_ARCHITECTURE, lambda: self.visualizer.visualize(nn.Linear(2, 2)))
        self.assert_code(constants.ERROR_UNKNOWN_BACKEND, lambda: self.visualizer.visualize(self.model, DiagramConfig(backend="torchview")))
        self.assert_code(constants.ERROR_UNSUPPORTED_FORMAT, lambda: self.visualizer.visualize(self.model, DiagramConfig(format="svg")))
        self.assert_code(constants.ERROR_UNSUPPORTED_VIEW, lambda: self.visualizer.visualize(self.model, DiagramConfig(views=(constants.VIEW_EXECUTION,))))
        self.assert_code(constants.ERROR_UNSUPPORTED_VIEW, lambda: self.visualizer.visualize(self.model, DiagramConfig(scope=constants.BIDIRECTIONAL_ALL)))

    def test_config_validation(self):
        for kwargs in [
            {constants.FIELD_VIEWS: ()}, {constants.FIELD_VIEWS: ("wrong",)}, {constants.FIELD_MAX_NODES: 0},
            {constants.FIELD_MAX_MODULES: True}, {constants.FIELD_LAYER_INDEX: -1}, {constants.FIELD_DIRECTION: 'LR; evil'},
        ]:
            self.assert_code(constants.ERROR_INVALID_CONFIGURATION, lambda: DiagramConfig(**kwargs))
        self.assert_code(constants.ERROR_INVALID_CONFIGURATION, lambda: self.visualizer.visualize(
            self.model, DiagramConfig(views=(constants.VIEW_DECODER,), layer_index=35)))

    def test_limits_and_version(self):
        self.assert_code(constants.ERROR_INSPECTION_LIMIT, lambda: self.visualizer.visualize(
            self.model, DiagramConfig(max_modules=2)))
        self.assert_code(constants.ERROR_RENDER_LIMIT, lambda: self.visualizer.visualize(
            self.model, DiagramConfig(max_nodes=2)))
        with patch("llm_graph.gemma4.version", return_value="5.19.1"):
            self.assert_code(constants.ERROR_UNSUPPORTED_TRANSFORMERS_VERSION, lambda: self.visualizer.visualize(self.model))

    def test_custom_forward_and_dimensions_rejected(self):
        layer = self.model.model.language_model.layers[0]
        with patch.object(layer, constants.METHOD_FORWARD, lambda *args: None):
            self.assert_code(constants.ERROR_ARCHITECTURE_LAYOUT, lambda: self.visualizer.visualize(self.model))
        with patch.object(layer.self_attn.q_proj, constants.FIELD_OUT_FEATURES, 42):
            self.assert_code(constants.ERROR_ARCHITECTURE_LAYOUT, lambda: self.visualizer.visualize(self.model))
        with patch.object(layer.self_attn.q_proj, constants.FIELD_NUM_BITS, 2):
            self.assert_code(constants.ERROR_PROPERTY_CONFLICT, lambda: self.visualizer.visualize(self.model))

    def test_unquantized_config_only_widths(self):
        c = Gemma4Config(**json.loads(PROFILE.read_text()))
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(c)
        report = self.visualizer.visualize(model)
        self.assertTrue(any(d.code == constants.WARNING_CONFIGURED_QUANTIZATION for d in report.diagnostics))

    def test_factories_are_isolated(self):
        from llm_graph.gemma4 import Gemma4Adapter
        duplicate = create_visualizer(adapters=[Gemma4Adapter(), Gemma4Adapter()])
        self.assert_code(constants.ERROR_AMBIGUOUS_ADAPTER, lambda: duplicate.visualize(self.model))
        empty = create_visualizer(adapters=[])
        self.assert_code(constants.ERROR_UNSUPPORTED_ARCHITECTURE, lambda: empty.visualize(self.model))
        self.assertEqual(len(self.visualizer.adapters.adapters), 1)

    def test_catalog_and_output(self):
        directory = (ROOT / "ERRORS.md").read_text()
        for code in CATALOG:
            self.assertIn(code, directory)
        with tempfile.TemporaryDirectory() as folder:
            paths = self.report.write(folder)
            self.assertEqual(len(paths), 7)
            data = json.loads((Path(folder) / constants.REPORT_JSON).read_text())
            self.assertEqual(data[constants.FIELD_SCHEMA_VERSION], 1)
            self.assert_code(constants.ERROR_ARTIFACT_EXISTS, lambda: self.report.write(folder))
            self.report.write(folder, overwrite=True)
            self.assertEqual(set(p.name for p in Path(folder).iterdir()), {p.name for p in paths})
            with patch("os.link", side_effect=PermissionError):
                self.assert_code(constants.ERROR_ARTIFACT_WRITE, lambda: self.report.write(Path(folder) / "denied"))

    def test_decoder_inventory_has_contiguous_rows(self):
        def assert_table(markdown):
            section = markdown.split("## Decoder inventory\n\n", 1)[1].split(
                "\n## Diagnostics", 1
            )[0].strip()
            lines = section.splitlines()
            self.assertEqual(len(lines), 37)  # Header, separator, 35 decoder rows.
            self.assertTrue(all(line.startswith(constants.REGEX_ALTERNATION) and line.endswith(constants.REGEX_ALTERNATION) for line in lines))
            self.assertEqual(
                [int(line.split(constants.REGEX_ALTERNATION)[1].strip()) for line in lines[2:]],
                list(range(35)),
            )

        assert_table(self.report.markdown())
        with tempfile.TemporaryDirectory() as folder:
            self.report.write(folder)
            assert_table((Path(folder) / constants.REPORT_MARKDOWN).read_text())

    def test_aliases_and_cycles_terminate(self):
        parent = nn.Module()
        child = nn.Linear(2, 2)
        parent.a = child
        parent.b = child
        child.loop = parent
        inventory = ModuleInspector().inspect(parent, 20)
        self.assertLess(len(inventory), 6)
        self.assertTrue(any(m.alias_of == "a" for m in inventory))
        self.assertTrue(any(m.alias_of == constants.EMPTY for m in inventory))

    def test_graph_validation_and_escaping(self):
        from dataclasses import replace
        bad = GraphView("bad", "bad", (Node("a", "a", constants.KIND_INPUT),), (Edge("a", "missing"),))
        self.assert_code(constants.ERROR_INVALID_GRAPH, lambda: ViewBuilder().build(
            replace(self.report.graph, overview=bad), DiagramConfig(views=(constants.VIEW_OVERVIEW,))))
        self.assertNotIn('"', escape_label('"]; click x "https://example.com"'))

    def test_import_from_other_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            import sys
            result = subprocess.run(
                [sys.executable, "-c", "from llm_graph import create_visualizer; create_visualizer()"],
                cwd=folder, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mermaid_parser(self):
        if not (ROOT / "node_modules/mermaid").exists():
            self.skipTest("Install documented local Mermaid validation dependencies")
        with tempfile.TemporaryDirectory() as folder:
            for artifact in self.report.artifacts:
                text = artifact.content.split("```mermaid\n", 1)[1].split("```", 1)[0]
                (Path(folder) / (artifact.name + ".mmd")).write_text(text)
            hierarchy = self.visualizer.visualize(self.model, DiagramConfig(
                views=(constants.VIEW_HIERARCHY,), format=constants.BACKEND_MERMAID, direction=constants.DIRECTION_LEFT_RIGHT, max_nodes=5000,
            ))
            (Path(folder) / "hierarchy.mmd").write_text(hierarchy.artifacts[0].content)
            result = subprocess.run(
                ["node", str(ROOT / "tests/parse_mermaid.mjs"), folder],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
